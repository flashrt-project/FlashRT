"""Phase-B de-risk — frt_graph adopts a REAL Qwen3.6 decode graph.

Proves the contract on the real model with minimal blast radius (no frontend
edits): capture stays in torch.cuda.graph (allocator-safe, byte-identical),
frt adopts torch's instantiated graph-exec via raw_cuda_graph_exec(), and the
exec layer drives replay on a wrapped torch stream. Assert the decode logits
produced by frt-driven replay are BIT-IDENTICAL to torch's own g.replay().

Run (inside the CUDA container):
    PYTHONPATH=.:./exec/build \
    FLASHRT_QWEN36_NVFP4_CKPT_DIR=checkpoints/qwen36_nvfp4 \
    FLASHRT_QWEN36_MTP_CKPT_DIR=checkpoints/qwen36_mtp_inferrouter \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python exec/tests/test_adopt_qwen36.py
"""

import os
import torch
import _flashrt_exec as ex

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

CKPT = os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR",
                      "checkpoints/qwen36_nvfp4")
CUR_POS = 0


def _snapshot(fe, cur_pos):
    return {
        "lin": fe._lin_state.clone(),
        "conv": fe._lin_conv_state.clone(),
        "k": fe._attn.K_cache[:, cur_pos:cur_pos + 1].clone(),
        "v": fe._attn.V_cache[:, cur_pos:cur_pos + 1].clone(),
    }


def _restore(fe, cur_pos, snap):
    fe._lin_state.copy_(snap["lin"])
    fe._lin_conv_state.copy_(snap["conv"])
    fe._attn.K_cache[:, cur_pos:cur_pos + 1].copy_(snap["k"])
    fe._attn.V_cache[:, cur_pos:cur_pos + 1].copy_(snap["v"])


def main():
    assert torch.cuda.is_available()
    fe = Qwen36TorchFrontendRtx(CKPT, device="cuda:0", max_seq=2048, quant="nvfp4")

    ids = fe._tokenizer("Explain quantum entanglement.", return_tensors="pt").input_ids.cuda()
    fe.reset_state()
    fe.reset_mtp_state()
    if not hasattr(fe, "_rope_cos_table"):
        fe._build_rope_table()
    fe._static_token_id.copy_(ids[:, 0:1])
    cos, sin = fe._rope_cos_sin(CUR_POS)

    # torch captures the decode graph (this is the production capture path).
    g = fe._ensure_graph_for_pos_nvfp4(CUR_POS)
    gs = fe._graph_stream

    # State after capture == pre-step (the method snaps/restores). Snapshot it
    # so both replays start identically.
    snap = _snapshot(fe, CUR_POS)

    def replay_torch():
        _restore(fe, CUR_POS, snap)
        fe._logits_buf.zero_()
        torch.cuda.synchronize()
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            g.replay()
        torch.cuda.current_stream().wait_stream(gs)
        torch.cuda.synchronize()
        return fe._logits_buf.clone()

    # frt adopts torch's instantiated exec and replays on the wrapped torch stream.
    ctx = ex.Ctx()
    gs_id = ctx.wrap_stream(gs.cuda_stream)
    fg = ctx.graph("decode", max_variants=256)
    fg.adopt(CUR_POS, g.raw_cuda_graph_exec())

    def replay_frt():
        _restore(fe, CUR_POS, snap)
        fe._logits_buf.zero_()
        torch.cuda.synchronize()
        gs.wait_stream(torch.cuda.current_stream())
        rc = fg.replay(CUR_POS, gs_id)
        assert rc == 0, f"frt replay rc={rc}"
        torch.cuda.current_stream().wait_stream(gs)
        torch.cuda.synchronize()
        return fe._logits_buf.clone()

    logits_torch = replay_torch()
    logits_frt = replay_frt()
    # cross-check torch is itself reproducible from the same start
    logits_torch2 = replay_torch()

    bit_identical = torch.equal(logits_torch, logits_frt)
    torch_repro = torch.equal(logits_torch, logits_torch2)
    max_abs = (logits_torch.float() - logits_frt.float()).abs().max().item()

    print("\n===== ADOPT REAL QWEN3.6 DECODE GRAPH =====")
    print(f"cur_pos              : {CUR_POS}")
    print(f"logits shape         : {tuple(logits_frt.shape)}")
    print(f"torch self-reproduce : {torch_repro}")
    print(f"frt == torch (exact) : {bit_identical}")
    print(f"max |frt - torch|    : {max_abs:.3e}")
    print(f"frt argmax           : {int(logits_frt.argmax())}")
    print(f"torch argmax         : {int(logits_torch.argmax())}")
    assert torch_repro, "torch replay not reproducible from restored state"
    assert bit_identical, "frt-driven replay diverged from torch replay"
    print("\nPASS — frt adopt+replay is BIT-IDENTICAL to torch on a real Qwen3.6 decode graph")


if __name__ == "__main__":
    main()
