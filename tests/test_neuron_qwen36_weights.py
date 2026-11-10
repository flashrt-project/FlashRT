"""Derived from KevGomes1403/nki-moe-megakernel @ 5879c39 (Apache-2.0).

The pinned implementation uses transposed MTP linear weights. Validate their
linear-map semantics; upstream shape assertions predate that layout change.

CPU unit tests for the A3B HF-to-NxDI weight converter.

Validates:
  * router rename            mlp.gate -> mlp.moe.router.linear_router
  * routed expert fusion     256 x {gate,up,down}_proj  ->
                             mlp.moe.expert_mlps.mlp_op.{gate_up_proj,down_proj}
  * shared expert pass-through (singular name, owned by NeuronMoEBlock)
  * shared_expert_gate pass-through (sigmoid gate over shared output)
  * MTP key remap            mtp.fc/pre_fc_norm_*/norm + mtp.layers.0.* -> mtp_head.*
  * MTP draft layer gets the same full-attention + MoE conversions as a main layer
  * MTP RMSNorms converted to (1 + w); draft LM head tied to the main lm_head
"""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("neuronx_distributed_inference")
import torch

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig  # noqa: E402

from flash_rt.neuron._vendor.qwen36.modeling_qwen36_a3b import (  # noqa: E402
    NeuronQwen36A3BForCausalLM,
    Qwen36A3BInferenceConfig,
    convert_qwen36_a3b_hf_to_neuron_state_dict,
    reorder_deltanet_qkv_for_tp,
)


# ---------------------------------------------------------------------------
# Fixture builders -- a deliberately tiny A3B-shaped model.
# ---------------------------------------------------------------------------


def _make_mini_config(
    num_layers: int = 4, num_experts: int = 4
) -> Qwen36A3BInferenceConfig:
    nc_cls = Qwen36A3BInferenceConfig.get_neuron_config_cls()
    nc = nc_cls(
        tp_degree=2,
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    )
    cfg = Qwen36A3BInferenceConfig(
        neuron_config=nc,
        hidden_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000,
        hidden_act="silu",
        # DeltaNet (tiny)
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        # MoE (tiny)
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        norm_topk_prob=True,
        # MTP
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
    )
    return cfg


def _make_mini_state_dict(cfg: Qwen36A3BInferenceConfig) -> dict:
    sd: dict = {}
    H = cfg.hidden_size
    V = cfg.vocab_size
    I = cfg.moe_intermediate_size
    SI = cfg.shared_expert_intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    bf = torch.bfloat16
    sd["embed_tokens.weight"] = torch.randn(V, H, dtype=bf) * 0.02
    sd["lm_head.weight"] = torch.randn(V, H, dtype=bf) * 0.02
    sd["norm.weight"] = torch.zeros(H, dtype=bf)  # +1 convention

    for l in range(cfg.num_hidden_layers):
        sd[f"layers.{l}.input_layernorm.weight"] = torch.zeros(H, dtype=bf)
        sd[f"layers.{l}.post_attention_layernorm.weight"] = torch.zeros(H, dtype=bf)

        # --- MoE replaces dense MLP. HF ships routed experts already fused
        #     into 3D tensors per layer (gate_up_proj: (E, 2I, H);
        #     down_proj: (E, H, I)). Our converter renames + transposes.
        sd[f"layers.{l}.mlp.gate.weight"] = (
            torch.randn(cfg.num_experts, H, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.experts.gate_up_proj"] = (
            torch.randn(cfg.num_experts, 2 * I, H, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.experts.down_proj"] = (
            torch.randn(cfg.num_experts, H, I, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.shared_expert.gate_proj.weight"] = (
            torch.randn(SI, H, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.shared_expert.up_proj.weight"] = (
            torch.randn(SI, H, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.shared_expert.down_proj.weight"] = (
            torch.randn(H, SI, dtype=bf) * 0.02
        )
        sd[f"layers.{l}.mlp.shared_expert_gate.weight"] = (
            torch.randn(1, H, dtype=bf) * 0.02
        )

        if cfg.layer_types[l] == "full_attention":
            # GQA: q_proj is doubled (query + output gate, interleaved per head).
            sd[f"layers.{l}.self_attn.q_proj.weight"] = (
                torch.randn(num_heads * head_dim * 2, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.k_proj.weight"] = (
                torch.randn(num_kv * head_dim, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.v_proj.weight"] = (
                torch.randn(num_kv * head_dim, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.o_proj.weight"] = (
                torch.randn(H, num_heads * head_dim, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.self_attn.q_norm.weight"] = torch.zeros(head_dim, dtype=bf)
            sd[f"layers.{l}.self_attn.k_norm.weight"] = torch.zeros(head_dim, dtype=bf)
        else:
            key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
            value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            conv_dim = key_dim * 2 + value_dim
            sd[f"layers.{l}.linear_attn.in_proj_qkv.weight"] = (
                torch.randn(conv_dim, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.linear_attn.in_proj_z.weight"] = (
                torch.randn(value_dim, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.linear_attn.in_proj_a.weight"] = (
                torch.randn(cfg.linear_num_value_heads, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.linear_attn.in_proj_b.weight"] = (
                torch.randn(cfg.linear_num_value_heads, H, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.linear_attn.conv1d.weight"] = (
                torch.randn(conv_dim, 1, cfg.linear_conv_kernel_dim, dtype=bf) * 0.02
            )
            sd[f"layers.{l}.linear_attn.A_log"] = torch.randn(
                cfg.linear_num_value_heads, dtype=bf
            )
            sd[f"layers.{l}.linear_attn.dt_bias"] = torch.randn(
                cfg.linear_num_value_heads, dtype=bf
            )
            sd[f"layers.{l}.linear_attn.norm.weight"] = (
                torch.randn(value_dim, dtype=bf) * 0.5
            )
            sd[f"layers.{l}.linear_attn.out_proj.weight"] = (
                torch.randn(H, value_dim, dtype=bf) * 0.02
            )

    # --- MTP head (real Qwen3-Next layout). The draft layer is a standard
    #     full-attention + MoE layer; embeddings/LM head are not shipped (tied). ---
    sd["mtp.fc.weight"] = torch.randn(H, 2 * H, dtype=bf) * 0.02
    sd["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(H, dtype=bf)
    sd["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(H, dtype=bf)
    sd["mtp.norm.weight"] = torch.zeros(H, dtype=bf)

    mp = "mtp.layers.0"
    sd[f"{mp}.input_layernorm.weight"] = torch.zeros(H, dtype=bf)
    sd[f"{mp}.post_attention_layernorm.weight"] = torch.zeros(H, dtype=bf)
    sd[f"{mp}.self_attn.q_proj.weight"] = (
        torch.randn(num_heads * head_dim * 2, H, dtype=bf) * 0.02
    )
    sd[f"{mp}.self_attn.k_proj.weight"] = (
        torch.randn(num_kv * head_dim, H, dtype=bf) * 0.02
    )
    sd[f"{mp}.self_attn.v_proj.weight"] = (
        torch.randn(num_kv * head_dim, H, dtype=bf) * 0.02
    )
    sd[f"{mp}.self_attn.o_proj.weight"] = (
        torch.randn(H, num_heads * head_dim, dtype=bf) * 0.02
    )
    sd[f"{mp}.self_attn.q_norm.weight"] = torch.zeros(head_dim, dtype=bf)
    sd[f"{mp}.self_attn.k_norm.weight"] = torch.zeros(head_dim, dtype=bf)
    sd[f"{mp}.mlp.gate.weight"] = torch.randn(cfg.num_experts, H, dtype=bf) * 0.02
    sd[f"{mp}.mlp.experts.gate_up_proj"] = (
        torch.randn(cfg.num_experts, 2 * I, H, dtype=bf) * 0.02
    )
    sd[f"{mp}.mlp.experts.down_proj"] = (
        torch.randn(cfg.num_experts, H, I, dtype=bf) * 0.02
    )
    sd[f"{mp}.mlp.shared_expert.gate_proj.weight"] = torch.randn(SI, H, dtype=bf) * 0.02
    sd[f"{mp}.mlp.shared_expert.up_proj.weight"] = torch.randn(SI, H, dtype=bf) * 0.02
    sd[f"{mp}.mlp.shared_expert.down_proj.weight"] = torch.randn(H, SI, dtype=bf) * 0.02
    sd[f"{mp}.mlp.shared_expert_gate.weight"] = torch.randn(1, H, dtype=bf) * 0.02

    return sd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeltaNetInProjFusion(unittest.TestCase):
    def test_fused_in_proj_and_transposed_out_proj(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        tp = cfg.neuron_config.tp_degree
        H = cfg.hidden_size
        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        nv = cfg.linear_num_value_heads
        fused_dim = conv_dim + value_dim + 2 * nv

        # Capture originals -- conversion pops the four source keys in place.
        orig = {}
        for l in range(cfg.num_hidden_layers):
            if cfg.layer_types[l] == "full_attention":
                continue
            p = f"layers.{l}.linear_attn"
            orig[l] = {
                "qkv": sd[f"{p}.in_proj_qkv.weight"].clone(),
                "z": sd[f"{p}.in_proj_z.weight"].clone(),
                "a": sd[f"{p}.in_proj_a.weight"].clone(),
                "b": sd[f"{p}.in_proj_b.weight"].clone(),
                "out": sd[f"{p}.out_proj.weight"].clone(),
            }
        self.assertTrue(orig, "fixture has no linear_attention layers")

        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)

        cd_l, vd_l, nv_l = conv_dim // tp, value_dim // tp, nv // tp
        il = fused_dim // tp
        for l, o in orig.items():
            p = f"layers.{l}.linear_attn"
            for nm in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                self.assertNotIn(f"{p}.{nm}.weight", out)

            fused = out[f"{p}.in_proj_fused.weight"]
            self.assertEqual(fused.shape, (H, fused_dim))

            # o_proj stored transposed: [hidden, value_dim] -> [value_dim, hidden].
            self.assertEqual(out[f"{p}.out_proj.weight"].shape, (value_dim, H))
            self.assertTrue(
                torch.equal(out[f"{p}.out_proj.weight"], o["out"].t().contiguous())
            )

            # Each rank's contiguous column block is [qkv_r | z_r | a_r | b_r].
            qkv_r = o["qkv"]
            if tp > 1:
                qkv_r = reorder_deltanet_qkv_for_tp(
                    qkv_r,
                    tp,
                    cfg.linear_num_key_heads,
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                )
            rows = fused.t()  # [I, hidden]
            for r in range(tp):
                blk = rows[r * il : (r + 1) * il]
                off = 0
                self.assertTrue(
                    torch.equal(blk[off : off + cd_l], qkv_r[r * cd_l : (r + 1) * cd_l])
                )
                off += cd_l
                self.assertTrue(
                    torch.equal(
                        blk[off : off + vd_l], o["z"][r * vd_l : (r + 1) * vd_l]
                    )
                )
                off += vd_l
                self.assertTrue(
                    torch.equal(
                        blk[off : off + nv_l], o["a"][r * nv_l : (r + 1) * nv_l]
                    )
                )
                off += nv_l
                self.assertTrue(
                    torch.equal(
                        blk[off : off + nv_l], o["b"][r * nv_l : (r + 1) * nv_l]
                    )
                )


class TestRouterRename(unittest.TestCase):
    def test_gate_to_linear_router(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            self.assertNotIn(f"layers.{l}.mlp.gate.weight", out)
            self.assertIn(f"layers.{l}.mlp.moe.router.linear_router.weight", out)
            self.assertEqual(
                out[f"layers.{l}.mlp.moe.router.linear_router.weight"].shape,
                (cfg.num_experts, cfg.hidden_size),
            )


class TestRoutedExpertFusion(unittest.TestCase):
    def test_fused_shapes(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            gu = out[f"layers.{l}.mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight"]
            dp = out[f"layers.{l}.mlp.moe.expert_mlps.mlp_op.down_proj.weight"]
            self.assertEqual(
                gu.shape,
                (cfg.num_experts, cfg.hidden_size, 2 * cfg.moe_intermediate_size),
            )
            self.assertEqual(
                dp.shape, (cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size)
            )

    def test_fused_hf_keys_removed(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            self.assertNotIn(f"layers.{l}.mlp.experts.gate_up_proj", out)
            self.assertNotIn(f"layers.{l}.mlp.experts.down_proj", out)

    def test_transpose_values_preserved(self):
        """NxDI gate_up_proj should be HF gate_up_proj transposed on dims (1, 2)."""
        cfg = _make_mini_config(num_layers=4, num_experts=2)
        sd = _make_mini_state_dict(cfg)
        orig_gu = sd["layers.0.mlp.experts.gate_up_proj"].clone()
        orig_dp = sd["layers.0.mlp.experts.down_proj"].clone()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        gu = out["layers.0.mlp.moe.expert_mlps.mlp_op.gate_up_proj.weight"]
        dp = out["layers.0.mlp.moe.expert_mlps.mlp_op.down_proj.weight"]
        torch.testing.assert_close(gu, orig_gu.transpose(1, 2).contiguous())
        torch.testing.assert_close(dp, orig_dp.transpose(1, 2).contiguous())


class TestSharedExpert(unittest.TestCase):
    def test_pass_through(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            for suf in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                self.assertIn(f"layers.{l}.mlp.shared_expert.{suf}", out)

    def test_shared_expert_gate_pass_through(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        for l in range(cfg.num_hidden_layers):
            self.assertIn(f"layers.{l}.mlp.shared_expert_gate.weight", out)


class TestMTPRemap(unittest.TestCase):
    def test_top_level_rename(self):
        cfg = _make_mini_config()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(
            _make_mini_state_dict(cfg), cfg
        )
        for old in (
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
        ):
            self.assertNotIn(old, out)
        self.assertIn("mtp_head.embed_norm.weight", out)
        self.assertIn("mtp_head.hidden_norm.weight", out)
        self.assertIn("mtp_head.final_norm.weight", out)
        self.assertEqual(
            out["mtp_head.eh_proj.weight"].shape, (2 * cfg.hidden_size, cfg.hidden_size)
        )

    def test_no_stray_mtp_keys(self):
        cfg = _make_mini_config()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(
            _make_mini_state_dict(cfg), cfg
        )
        self.assertEqual([k for k in out if k.startswith("mtp.")], [])

    def test_draft_layer_full_attention_converted(self):
        """The MTP decoder layer gets the same GQA conversions as a main layer."""
        cfg = _make_mini_config()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(
            _make_mini_state_dict(cfg), cfg
        )
        p = "mtp_head.decoder_layer.self_attn"
        # q_proj split into query + sigmoid output gate; q/k norms renamed.
        self.assertEqual(
            out[f"{p}.q_proj.weight"].shape,
            (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
        )
        self.assertIn(f"{p}.output_gate_proj.weight", out)
        self.assertIn(f"{p}.q_layernorm.weight", out)
        self.assertIn(f"{p}.k_layernorm.weight", out)
        self.assertNotIn(f"{p}.q_norm.weight", out)

    def test_draft_layer_moe_converted(self):
        cfg = _make_mini_config()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(
            _make_mini_state_dict(cfg), cfg
        )
        p = "mtp_head.decoder_layer.mlp"
        self.assertIn(f"{p}.moe.router.linear_router.weight", out)
        self.assertEqual(
            out[f"{p}.moe.expert_mlps.mlp_op.gate_up_proj.weight"].shape,
            (cfg.num_experts, cfg.hidden_size, 2 * cfg.moe_intermediate_size),
        )
        self.assertIn(f"{p}.shared_expert_gate.weight", out)

    def test_norms_one_plus_w(self):
        """Zero-init HF norms become 1.0 after the (1 + w) conversion."""
        cfg = _make_mini_config()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(
            _make_mini_state_dict(cfg), cfg
        )
        for nk in (
            "mtp_head.embed_norm.weight",
            "mtp_head.hidden_norm.weight",
            "mtp_head.final_norm.weight",
            "mtp_head.decoder_layer.input_layernorm.weight",
        ):
            torch.testing.assert_close(
                out[nk].float(), torch.ones_like(out[nk].float())
            )

    def test_lm_head_tied_to_main(self):
        cfg = _make_mini_config()
        sd = _make_mini_state_dict(cfg)
        main_lm_head = sd["lm_head.weight"].clone()
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(sd, cfg)
        self.assertIn("mtp_head.mtp_lm_head.weight", out)
        x = torch.randn(2, cfg.hidden_size, dtype=main_lm_head.dtype)
        torch.testing.assert_close(x @ out["mtp_head.mtp_lm_head.weight"], torch.nn.functional.linear(x, main_lm_head))
        torch.testing.assert_close(
            out["mtp_head.mtp_lm_head.weight"], out["lm_head.weight"]
        )

    def test_disabled_drops_mtp(self):
        cfg = _make_mini_config()
        cfg.mtp_num_hidden_layers = 0
        out = convert_qwen36_a3b_hf_to_neuron_state_dict(
            _make_mini_state_dict(cfg), cfg
        )
        self.assertEqual([k for k in out if k.startswith("mtp")], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
