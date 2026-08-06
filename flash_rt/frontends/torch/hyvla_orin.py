"""HyVLA frontend for Jetson Orin SM87.

SM87 has no native FP8/FP4 tensor cores, so the Thor FP8/NVFP4 paths are
mapped to the existing SM80-family INT8 W8A8 rowwise kernels. The public IO,
preprocessing, prefix assembly, graph cache, and weight spec are inherited from
``HyVLATorchFrontendThor``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.frontends.torch.hyvla_thor import HyVLATorchFrontendThor, _BF16
from flash_rt.models.hyvla.pipeline_orin import HyVLAOrinBF16Pipeline

logger = logging.getLogger(__name__)

INT8_QUANT_MAX = 127.0
INT8_QUANT_EPS = 1e-12


def _quantize_per_row_int8(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w_f32 = w_bf16.float().contiguous()
    scale = torch.clamp(
        w_f32.abs().amax(dim=1) / INT8_QUANT_MAX,
        min=INT8_QUANT_EPS,
    ).to(device=w_bf16.device, dtype=torch.float32).contiguous()
    q = torch.clamp(
        torch.round(w_f32 / scale[:, None]),
        -127, 127,
    ).to(torch.int8).contiguous()
    return q, scale


class HyVLATorchFrontendOrin(HyVLATorchFrontendThor):
    def __init__(self, checkpoint_dir: str, *, hardware: str = "rtx_sm87",
                 use_fp8: bool = True, use_fp8_vit: bool = False,
                 use_fused: bool = True, use_fp4: bool = False,
                 use_fused_quant: bool = False, use_autotune: bool = False,
                 use_ffn_mega: bool = False, use_int8: bool | None = None,
                 use_int8_vlm: bool = False,  # experimental: not validated at cosine >= 0.999
                 use_int8_vlm_ffn: bool | None = None,  # experimental: not validated at cosine >= 0.999
                 use_int8_exp: bool = True,  # experimental: not validated at cosine >= 0.999
                 use_int8_vit: bool | None = None,  # experimental: not validated at cosine >= 0.999
                 vit_int8_parts: tuple = ("qkv", "proj", "fc1", "fc2"),  # experimental: not validated at cosine >= 0.999
                 **kwargs):
        if use_fp4:
            raise RuntimeError(
                "HyVLATorchFrontendOrin does not support FP4: SM87 has no "
                "native FP4 tensor cores. Use the default INT8 W8A8 path or "
                "pass use_fp8=False, use_int8=False for BF16.")
        if use_fp8_vit:
            logger.warning(
                "HyVLA Orin ignores Thor's use_fp8_vit; ViT quantization on "
                "SM87 uses the INT8 path (use_int8_vit).")
        if use_fused_quant:
            logger.warning("HyVLA Orin disables Thor-only fused FP8 quantization.")
        if use_ffn_mega:
            logger.warning("HyVLA Orin disables Thor-only FP8 FFN megakernels.")
        if use_autotune:
            logger.warning("HyVLA Orin INT8 path does not use Thor FP8 autotune.")

        self.use_int8 = bool(use_fp8) if use_int8 is None else bool(use_int8)
        self.use_int8_vlm = bool(use_int8_vlm)
        self.use_int8_exp = bool(use_int8_exp)
        self.use_int8_vit = False if use_int8_vit is None else bool(use_int8_vit)
        self.vit_int8_parts = tuple(vit_int8_parts)
        vlm_ffn = self.use_int8 if use_int8_vlm_ffn is None else bool(use_int8_vlm_ffn)
        if use_fp8 and self.use_int8:
            logger.info(
                "HyVLA Orin maps use_fp8=True to SM87 INT8 W8A8 rowwise GEMMs "
                "(expert tower + prefill FFN by default; use_int8_vlm=True "
                "quantizes prefill QKV/O too, use_int8_vit=True is opt-in and "
                "fails the 0.999 E2E gate).")

        super().__init__(
            checkpoint_dir,
            hardware=hardware,
            use_fp8=False,
            use_fp8_vit=False,
            use_fused=False,
            use_fp4=False,
            use_fused_quant=False,
            use_autotune=False,
            use_ffn_mega=False,
            **kwargs,
        )
        if self.use_int8_vit:
            self._quantize_vit_int8()
        self.pipe = HyVLAOrinBF16Pipeline(self)
        import flash_rt.flash_rt_kernels as fvk
        # ViT levers: memory-efficient SDPA reads the strided QKV slices
        # directly (the flash backend force-copies them), and the fused
        # residual-add+LayerNorm kernel collapses the ViT elementwise pairs.
        self.pipe._vit_eff_sdpa = True
        self.pipe._vit_fuse_ln = hasattr(fvk, "hyvla_vit_add_layer_norm_bf16")
        if use_fused:
            if hasattr(fvk, "hyvla_rope_qknorm_kvwrite_bf16"):
                self.pipe._fused_attn = True
            else:
                logger.warning(
                    "HyVLA Orin: fused RoPE/QKNorm/KV-write kernel not in "
                    "this build; falling back to the torch attention-prep path.")
        if self.use_int8:
            self._quantize_int8()
            self._vlm_fp8_ready = self.use_int8_vlm
            self._exp_fp8_ready = self.use_int8_exp
            self.pipe.enable_int8()
            if vlm_ffn and not self.use_int8_vlm:
                self._quantize_vlm_ffn_int8()
                self.pipe._vlm_ffn_int8 = True

    def _quantize_int8(self):
        def q_list(src):
            qs, ss = [], []
            for w in src:
                q, s = _quantize_per_row_int8(w)
                qs.append(q)
                ss.append(s)
            return qs, ss

        if self.use_int8_exp:
            self._exp_qkv8, self._exp_qkv_ws = q_list(self._exp_qkv_v)
            self._exp_o8, self._exp_o_ws = q_list(self._exp_o_v)
            self._exp_gu8, self._exp_gu_ws = q_list(self._exp_gu_v)
            self._exp_d8, self._exp_d_ws = q_list(self._exp_d_v)
            self._exp_fp8_ready = True
        else:
            self._exp_fp8_ready = False

        if self.use_int8_vlm:
            self._vlm_qkv_v8, self._vlm_qkv_v_ws = q_list(self._vlm_qkv_v)
            self._vlm_o_v8, self._vlm_o_v_ws = q_list(self._vlm_o_v)
            self._vlm_gu_v8, self._vlm_gu_v_ws = q_list(self._vlm_gu_v)
            self._vlm_d_v8, self._vlm_d_v_ws = q_list(self._vlm_d_v)
            self._vlm_qkv_t8, self._vlm_qkv_t_ws = q_list(self._vlm_qkv_t)
            self._vlm_o_t8, self._vlm_o_t_ws = q_list(self._vlm_o_t)
            self._vlm_gu_t8, self._vlm_gu_t_ws = q_list(self._vlm_gu_t)
            self._vlm_d_t8, self._vlm_d_t_ws = q_list(self._vlm_d_t)
            self._vlm_fp8_ready = True
        else:
            self._vlm_fp8_ready = False
        torch.cuda.synchronize()

    def _quantize_vlm_ffn_int8(self):
        """INT8-quantize only the VLM prefill FFN weights (gate/up + down,
        vision and text branches). QKV/O stay BF16 so the KV cache written by
        prefill — and read by all 10 denoise steps — keeps full precision."""
        def q_list(src):
            qs, ss = [], []
            for w in src:
                q, s = _quantize_per_row_int8(w)
                qs.append(q)
                ss.append(s)
            return qs, ss

        self._vlm_gu_v8, self._vlm_gu_v_ws = q_list(self._vlm_gu_v)
        self._vlm_d_v8, self._vlm_d_v_ws = q_list(self._vlm_d_v)
        self._vlm_gu_t8, self._vlm_gu_t_ws = q_list(self._vlm_gu_t)
        self._vlm_d_t8, self._vlm_d_t_ws = q_list(self._vlm_d_t)
        torch.cuda.synchronize()
        logger.info(
            "HyVLA Orin: VLM prefill FFN (gate/up/down) quantized to INT8; "
            "QKV/O remain BF16.")

    def _quantize_vit_int8(self):
        """Replace the 27-block ViT GEMM weights (qkv/proj/fc1/fc2) in-place
        with per-row INT8 + FP32 scale wrappers consumed by
        ``HyVLAOrinBF16Pipeline._vit_*``. LayerNorm weights/biases and the
        patch-embed conv stay BF16."""
        from flash_rt.models.hyvla.pipeline_orin import _W8Int8

        part_attrs = {
            "qkv": "_vit_qkv_w",
            "proj": "_vit_proj_w",
            "fc1": "_vit_fc1_w",
            "fc2": "_vit_fc2_w",
        }
        for part in self.vit_int8_parts:
            if part not in part_attrs:
                raise ValueError(
                    f"unknown vit_int8_parts entry {part!r}; "
                    f"valid: {sorted(part_attrs)}")
            attr = part_attrs[part]
            src = getattr(self, attr)
            wrapped = []
            for w in src:
                q, s = _quantize_per_row_int8(w)
                wrapped.append(_W8Int8(q, s))
            setattr(self, attr, wrapped)
            del src
        torch.cuda.synchronize()
        logger.info(
            "HyVLA Orin: ViT INT8 W8A8 enabled for parts %s.",
            list(self.vit_int8_parts))


    @torch.no_grad()
    def predict_actions(self, images, prompt=None, state=None, noise=None,
                        use_graph=True):
        """Orin variant of the Thor ``predict_actions`` with a static-prefix
        cache: everything that depends only on (prompt, num_cam) — the segment
        mask, the [vision|text] permutation, the bf16-rounded RoPE tables
        (fp64 math), and the suffix mask — is computed once per prompt and
        reused across frames. Only the image-dependent ``prefix_embs`` are
        rebuilt each call."""
        if prompt is not None and prompt != self._prompt:
            self.set_prompt(prompt)
        assert self._lang_tokens is not None, "call set_prompt() first"
        dev = self.device

        if not torch.is_tensor(images):
            images = torch.as_tensor(np.asarray(images))
        cam_imgs = self._preprocess_images(images)
        imgs5 = torch.cat(cam_imgs, 0)
        merged = self._vit_merge(imgs5, use_graph=use_graph)

        if state is None:
            state_t = torch.zeros(1, self.max_state_dim, device=dev, dtype=_BF16)
        else:
            st = torch.as_tensor(np.asarray(state), device=dev, dtype=_BF16).reshape(1, -1)
            if st.shape[1] < self.max_state_dim:
                st = F.pad(st, (0, self.max_state_dim - st.shape[1]))
            state_t = st

        key = (self._prompt, merged.shape[0])
        stat = getattr(self, "_static_prefix_cache", None)
        if stat is None or stat["key"] != key:
            (prefix_embs, pad_masks, att_masks, mm_prefix,
             idx_ranges, full_ranges) = self._assemble_prefix(merged)
            att2d = self._make_att_2d(pad_masks, att_masks)
            att2d = self._apply_segment_mask(att2d, idx_ranges, full_ranges)
            prefix_pos = torch.cumsum(pad_masks.long(), dim=1) - 1
            mm = mm_prefix[0]
            perm = torch.cat([torch.nonzero(mm).squeeze(-1),
                              torch.nonzero(~mm).squeeze(-1)])
            n_vis = int(mm.sum().item())
            att2d = att2d[:, perm][:, :, perm]
            prefix_pos = prefix_pos[:, perm]
            pcos, psin = self._rope_cos_sin(prefix_pos)
            S_p = pad_masks.shape[1]
            S_s = 1 + self.chunk
            suffix_pad = torch.ones(1, S_s, dtype=torch.bool, device=dev)
            suffix_att = torch.tensor([1, 1] + [0] * (self.chunk - 1),
                                      dtype=torch.bool, device=dev)[None]
            suffix_att2d = self._make_att_2d(suffix_pad, suffix_att)
            prefix_pad_2d = pad_masks[:, perm][:, None, :].expand(1, S_s, S_p)
            smask = torch.cat([prefix_pad_2d, suffix_att2d], 2)[:, None]
            suffix_pos = (pad_masks.long().sum(-1)[:, None]
                          + torch.cumsum(suffix_pad.long(), 1) - 1)
            scos, ssin = self._rope_cos_sin(suffix_pos)
            stat = {"key": key, "perm": perm, "n_vis": n_vis, "S_p": S_p,
                    "pmask": att2d[:, None], "pcos": pcos, "psin": psin,
                    "smask": smask, "scos": scos, "ssin": ssin}
            self._static_prefix_cache = stat

        (prefix_embs, _pad_masks, _att_masks, _mm_prefix,
         _idx_ranges, _full_ranges) = self._assemble_prefix(merged)
        prefix_embs = prefix_embs[:, stat["perm"]]

        if noise is None:
            noise_t = torch.randn(1, self.chunk, self.max_action_dim,
                                  dtype=torch.float32, device=dev)
        else:
            noise_t = torch.as_tensor(np.asarray(noise), device=dev, dtype=torch.float32)

        x_t = self._graph_forward(stat["S_p"], stat["n_vis"], prefix_embs,
                                  stat["pmask"], stat["pcos"], stat["psin"],
                                  stat["smask"], stat["scos"], stat["ssin"],
                                  state_t, noise_t, use_graph=use_graph)
        return x_t.float().cpu().numpy()


__all__ = ["HyVLATorchFrontendOrin"]

