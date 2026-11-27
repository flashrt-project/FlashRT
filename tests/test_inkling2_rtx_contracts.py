"""CPU checks for the Inkling 2 Mini RTX path; no GPU, checkpoint or network required."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from flash_rt.frontends.torch import _inkling2_rtx_runtime as rtm  # noqa: E402
from flash_rt.frontends.torch._inkling2_rtx_mtp import MTP_GEOMETRY, check_mtp_config  # noqa: E402


def _config():
    text = dict(rtm.GEOMETRY, model_type="inkling2_loop_text", loop_config=dict(rtm.LOOP),
                local_layer_ids=[l for l in range(24) if l % 6 != 5], unpadded_vocab_size=200058)
    mtp = dict(MTP_GEOMETRY, rope_theta=1.0e6, target_layer_ids=[5, 23, 41, 48])
    return {"model_type": "inkling2_loop", "text_config": text, "mtp_config": mtp, "eos_token_id": 200006}


def test_routing_entry():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("inkling2", "torch", "rtx_sm120")] == (
        "flash_rt.frontends.torch.inkling2_rtx", "Inkling2TorchFrontendRtx")


def test_released_geometry_is_accepted():
    raw = _config()
    rtm.check_config(raw)
    assert check_mtp_config(raw, visits=48)["target_layer_ids"] == [5, 23, 41, 48]


@pytest.mark.parametrize("key,value", [("hidden_size", 4096), ("n_routed_experts", 64),
                                       ("sliding_window_size", 4096), ("n_shared_experts", 1)])
def test_other_geometry_is_rejected(key, value):
    raw = _config()
    raw["text_config"][key] = value
    with pytest.raises(rtm.Inkling2ConfigError, match=key):
        rtm.check_config(raw)


def test_loop_and_layer_layout_are_checked():
    raw = _config()
    raw["text_config"]["loop_config"]["num_passes"] = 2
    with pytest.raises(rtm.Inkling2ConfigError, match="num_passes"):
        rtm.check_config(raw)
    raw = _config()
    raw["text_config"]["local_layer_ids"] = list(range(24))
    with pytest.raises(rtm.Inkling2ConfigError, match="global layers"):
        rtm.check_config(raw)


def test_drafter_config_is_checked():
    raw = _config()
    raw["mtp_config"]["block_size"] = 16
    with pytest.raises(rtm.Inkling2ConfigError, match="block_size"):
        check_mtp_config(raw, visits=48)
    raw = _config()
    raw["mtp_config"]["target_layer_ids"] = [5, 23, 41, 47]
    with pytest.raises(rtm.Inkling2ConfigError, match="target_layer_ids"):
        check_mtp_config(raw, visits=48)
    with pytest.raises(rtm.Inkling2ConfigError, match="mtp_config"):
        check_mtp_config({"text_config": {}}, visits=48)


def test_read_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    raw, text = rtm.read_config(tmp_path)
    assert text["hidden_size"] == 2304 and raw["eos_token_id"] == 200006
    with pytest.raises(FileNotFoundError):
        rtm.read_config(tmp_path / "missing")


def test_checkpoint_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        rtm.checkpoint_shards(tmp_path)
    for name in ("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors", "mtp.safetensors"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in rtm.checkpoint_shards(tmp_path)] == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00002-of-00002.safetensors"}}))
    assert [p.name for p in rtm.checkpoint_shards(tmp_path)] == ["model-00002-of-00002.safetensors"]


@pytest.mark.parametrize("n", [1, 7, 8, 15, 16, 40, 511, 512, 513, 700, 1100, 4000])
@pytest.mark.parametrize("exact", [False, True])
def test_plan_chunks_covers_the_prompt(n, exact):
    sizes = (512, 256, 128, 64, 32, 16)
    plan = rtm.plan_chunks(n, sizes, 8, exact)
    assert sum(r for _, r in plan) == n
    kinds = [k for k, _ in plan]
    assert kinds == sorted(kinds, key=("chunk", "exact", "step").index)       # chunks, then exact, then steps
    assert kinds.count("step") < 8
    if exact:
        assert "chunk" not in kinds
    else:
        chunk = [r for k, r in plan if k == "chunk"]
        assert chunk == sorted(chunk, reverse=True) and all(r in sizes for r in chunk)
        assert kinds.count("exact") <= 1


def test_plan_chunks_rejects_an_empty_prompt():
    with pytest.raises(ValueError):
        rtm.plan_chunks(0, (512,), 8, False)


def test_visit_count():
    assert rtm.visit_count(_config()["text_config"]) == 48


def test_chat_template_tojson_matches_transformers():
    """tojson takes transformers' arguments and defaults (ensure_ascii=False): tool declarations render with sorted
    keys and compact separators as the checkpoint's template asks."""
    pytest.importorskip("jinja2")
    from flash_rt.frontends.torch.inkling2_rtx import compile_chat_template
    t = compile_chat_template('{{ x | tojson }}|{{ x | tojson(sort_keys=true, separators=(",", ":")) }}|'
                              '{{ x | tojson(indent=1) }}')
    x = {"b": 1, "a": "é"}
    assert t.render(x=x) == '{"b": 1, "a": "é"}|{"a":"é","b":1}|{\n "b": 1,\n "a": "é"\n}'
    with pytest.raises(ValueError, match="stop"):
        compile_chat_template("{{ raise_exception('stop') }}").render()
    assert compile_chat_template("{% generation %}A{% endgeneration %}B").render() == "AB"


def _released_tensors():
    formats = ["nvfp4"] * 20 + ["mxfp8"] * 4
    return rtm.expected_tensors(_config()["text_config"] | {"vocab_size": 201024}, formats)


def test_expected_tensors_follow_the_released_layout():
    t = _released_tensors()
    assert t["layers.0.mlp.experts.w13_weight"] == ("U8", (128, 2304, 1152))
    assert t["layers.0.mlp.experts.w2_weight_scale"] == ("F8_E4M3", (128, 2304, 72))
    assert t["layers.23.mlp.experts.w2_weight"] == ("F8_E4M3", (128, 2304, 1152))
    assert t["layers.23.mlp.experts.w13_weight_scale"] == ("U8", (128, 2304, 72))
    assert t["attn_res.attn_queries.4"] == ("F32", (3, 2304)) and t["attn_res.attn_queries.3"] == ("F32", (2304,))
    assert "layers.5.attn.k_sconv.static_w" in t and "layers.4.attn.k_sconv.weight" in t
    assert "layers.6.controllers.mlp.down_weight" in t and "layers.5.controllers.mlp.down_weight" not in t
    rtm.check_tensors(dict(t), t, "weights")
    assert rtm.expert_formats(t, 24) == ["nvfp4"] * 20 + ["mxfp8"] * 4


def test_bad_weights_are_refused_before_loading():
    t = _released_tensors()
    bad = dict(t)
    bad["layers.0.mlp.experts.w2_weight"] = ("F32", (1,))
    with pytest.raises(rtm.Inkling2ConfigError, match="layers.0.mlp.experts.w2_weight: F32 \\[1\\]"):
        rtm.check_tensors(bad, t, "weights")
    bad = dict(t)
    bad["layers.3.attn.wq_du.weight"] = ("BF16", (3072, 2048))
    with pytest.raises(rtm.Inkling2ConfigError, match="wq_du"):
        rtm.check_tensors(bad, t, "weights")
    short = {k: v for k, v in t.items() if k != "unembed.weight"}
    with pytest.raises(rtm.Inkling2ConfigError, match="1 missing \\(unembed.weight\\)"):
        rtm.check_tensors(short, t, "weights")
    with pytest.raises(rtm.Inkling2ConfigError, match="layer 2: experts stored as BF16"):
        rtm.expert_formats(dict(t, **{"layers.2.mlp.experts.w13_weight": ("BF16", (128, 2304, 2304))}), 24)


def test_drafter_tensors_are_checked():
    from flash_rt.frontends.torch._inkling2_rtx_mtp import expected_mtp_tensors
    t = expected_mtp_tensors(201024)
    assert t["markov_head.rnn_proj.weight"] == ("BF16", (768, 5120))
    assert t["layers.2.self_attn.k_proj_ctx.weight"] == ("BF16", (1024, 2304))
    with pytest.raises(rtm.Inkling2ConfigError, match="markov_w1"):
        rtm.check_tensors(dict(t, **{"markov_head.markov_w1.weight": ("BF16", (1, 256))}), t, "drafter weights")


def test_safetensors_header(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    path = tmp_path / "x.safetensors"
    st.save_file({"a": torch.zeros(3, 4, dtype=torch.bfloat16), "b": torch.zeros(1)}, str(path))
    assert rtm.safetensors_header(path) == {"a": ("BF16", (3, 4)), "b": ("F32", (1,))}
    (tmp_path / "bad.safetensors").write_bytes(b"\xff" * 16)
    with pytest.raises(rtm.Inkling2ConfigError):
        rtm.safetensors_header(tmp_path / "bad.safetensors")


def test_runtime_reads_no_torch_matmul_setting():
    """The products run on the module's cuBLAS handle; the runtime never reads or sets torch.backends."""
    from pathlib import Path
    for name in ("_inkling2_rtx_runtime.py", "_inkling2_rtx_mtp.py", "_inkling2_rtx_dflash2.py", "inkling2_rtx.py"):
        assert "torch.backends" not in (Path(rtm.__file__).parent / name).read_text()


def test_missing_extension_names_the_build_flag(monkeypatch):
    monkeypatch.setattr(rtm, "_ik", None)
    with pytest.raises(RuntimeError, match="FLASHRT_ENABLE_INKLING2"):
        rtm.Inkling2RtxRuntime("/nonexistent")


def _dflash2_dir(tmp_path, cfg=None, tensors=None):
    """A drafter directory: config.json and a safetensors file with the header only (the check reads no data)."""
    from flash_rt.frontends.torch._inkling2_rtx_dflash2 import DF_CONFIG, DF_GEOMETRY, expected_dflash2_tensors
    if cfg is None:
        cfg = dict(DF_GEOMETRY, rms_norm_eps=1e-6, rope_parameters={"rope_theta": 1e4},
                   dflash_config=dict(DF_CONFIG, mask_token_id=200056, final_logit_softcapping=30.0,
                                      target_layer_ids=[5, 23, 41, 48]))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    t = expected_dflash2_tensors(201024) if tensors is None else tensors
    header = {k: {"dtype": dt, "shape": list(shape), "data_offsets": [0, 0]} for k, (dt, shape) in t.items()}
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
    return cfg


def test_dflash2_drafter_is_checked(tmp_path):
    from flash_rt.frontends.torch._inkling2_rtx_dflash2 import check_dflash2, expected_dflash2_tensors
    text = {"vocab_size": 201024}
    cfg = _dflash2_dir(tmp_path)
    assert check_dflash2(tmp_path, text, 48)["num_hidden_layers"] == 5
    t = expected_dflash2_tensors(201024)
    assert t["fc.weight"] == ("BF16", (2304, 9216))
    assert t["layers.4.attention_conv.kernel_projection.weight"] == ("BF16", (576, 2304))
    for key, value, match in (("head_dim", 64, "head_dim"), ("sliding_window", 4096, "sliding_window")):
        _dflash2_dir(tmp_path, dict(cfg, **{key: value}))
        with pytest.raises(rtm.Inkling2ConfigError, match=match):
            check_dflash2(tmp_path, text, 48)
    for key, value in (("target_layer_ids", [5, 23, 41, 47]), ("block_size", 16), ("mask_token_id", 201024)):
        _dflash2_dir(tmp_path, dict(cfg, dflash_config=dict(cfg["dflash_config"], **{key: value})))
        with pytest.raises(rtm.Inkling2ConfigError, match=key):
            check_dflash2(tmp_path, text, 48)
    _dflash2_dir(tmp_path, dict(cfg, rope_parameters={}))
    with pytest.raises(rtm.Inkling2ConfigError, match="rope_theta"):
        check_dflash2(tmp_path, text, 48)
    _dflash2_dir(tmp_path, {k: v for k, v in cfg.items() if k != "dflash_config"})
    with pytest.raises(rtm.Inkling2ConfigError, match="dflash_config"):
        check_dflash2(tmp_path, text, 48)
    _dflash2_dir(tmp_path, cfg, dict(t, **{"layers.2.self_attn.k_proj.weight": ("BF16", (512, 2304))}))
    with pytest.raises(rtm.Inkling2ConfigError, match="k_proj"):
        check_dflash2(tmp_path, text, 48)


def test_dflash2_needs_speculative(tmp_path):
    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    with pytest.raises(ValueError, match="speculative=True"):
        Inkling2TorchFrontendRtx(str(tmp_path), dflash2=str(tmp_path))


class _Positions:
    """A runtime that only tracks positions: a decode step writes one row, a round of W rows writes W rows and
    accepts every draft (the farthest a round can move)."""

    def __init__(self, max_seq):
        import torch
        self.max_seq, self.reached, self.prefills, self.rounds = max_seq, 0, [], []
        self.pos = torch.zeros(1, dtype=torch.int32)
        self.history = torch.zeros(max_seq + 1, dtype=torch.int32)

    def reset(self):
        self.pos.zero_()

    def prefill(self, ids, exact=False):
        self.prefills.append(exact)
        self.pos.fill_(len(ids))
        return 0

    def run(self, name):
        rows = int(name[len("df_round"):]) if name.startswith("df_round") else 1
        if rows > 1:
            self.rounds.append((rows, int(self.drafter.log_count)))
        end = int(self.pos) + rows
        assert end <= self.max_seq, f"{name} writes position {end - 1} of max_seq {self.max_seq}"
        self.reached = max(self.reached, end)
        self.pos.fill_(end)


class _AdaptiveDrafter:
    adaptive, width, widths_all = True, 4, (2, 3, 4, 5, 6, 7, 8)

    def __init__(self, rt):
        import torch
        self.rt, self.log_count = rt, torch.zeros(1, dtype=torch.int32)
        rt.drafter = self

    def prefill(self, ids, exact=False):
        return self.rt.prefill(ids, exact=exact)

    def add_width(self, W):
        return f"df_round{W}"

    def steps(self, first, max_new_tokens, check_every=4):      # a request: rounds of 8 rows until the tokens
        self.log_count.fill_(1)
        done = 1
        while done < max_new_tokens:
            self.rt.run(self.add_width(8))
            done += 8
        yield [first] * max_new_tokens


@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_adaptive_warmup_stays_inside_max_seq(monkeypatch, mode):
    """Warmup's round of every width starts from the position after the prompt, in the frontend's prefill mode."""
    import torch
    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    fe = object.__new__(Inkling2TorchFrontendRtx)
    fe.rt = _Positions(2048)
    fe.drafter = _AdaptiveDrafter(fe.rt)
    fe.prefill_mode, fe.check_block, fe.check_rounds, fe.speculative = mode, 16, 1, True
    fe.vocab, fe.eog = 200064, set()
    fe.warmup(2022)                     # the longest prompt admitted: 2022 + 2 new + (1 + 1) * 8 + 8 = 2048
    assert {W for W, _ in fe.rt.rounds} == set(_AdaptiveDrafter.widths_all)
    assert all(log == 1 for _, log in fe.rt.rounds)           # each round's accepted tokens from the log's start
    assert fe.rt.reached <= 2048
    assert fe.rt.prefills and all(e == (mode == "exact") for e in fe.rt.prefills)
    with pytest.raises(ValueError, match="exceeds max_seq"):
        fe.warmup(2023)
