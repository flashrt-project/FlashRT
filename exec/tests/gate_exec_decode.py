"""Phase-B step 1 gate — decode-S=1 graph replay routed through the exec layer.

Runs the SAME process logic under FLASHRT_QWEN36_USE_EXEC=0 vs =1 (one value per
process) and prints:
  - output token sha (deterministic greedy correctness reference)
  - warm prefill TTFT (the decode-S=1 graph is the prefill path; this is where
    the wiring takes effect)

Gate (compared across the two runs by the caller):
  - sha MUST be identical  (frt-driven replay is bit-identical — proven in
    test_adopt_qwen36.py; this confirms it inside the real serving loop)
  - TTFT MUST NOT regress

Run twice (inside pi0-stablehlo-test):
  for v in 0 1; do
    PYTHONPATH=/workspace/PI/official/FlashRT-spec:/workspace/PI/official/FlashRT-spec/exec/build \
    FLASHRT_QWEN36_USE_EXEC=$v \
    FLASHRT_QWEN36_NVFP4_CKPT_DIR=/workspace/PI/checkpoints/qwen36_nvfp4 \
    FLASHRT_QWEN36_MTP_CKPT_DIR=/workspace/PI/checkpoints/qwen36_mtp_inferrouter \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python exec/tests/gate_exec_decode.py
  done
"""

import hashlib
import os
import torch

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

CKPT = os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR",
                      "/workspace/PI/checkpoints/qwen36_nvfp4")
USE_EXEC = os.environ.get("FLASHRT_QWEN36_USE_EXEC", "0")
PROMPT = "Explain quantum entanglement in one short paragraph."
K, N = 6, 128


def measure_ttft(fe, ids):
    prompt_len = int(ids.shape[1])
    hidden = fe._cfg["hidden_size"]
    fe.reset_state()
    fe.reset_mtp_state()
    if not hasattr(fe, "_rope_cos_table"):
        fe._build_rope_table()

    def prefill():
        fe.reset_state()
        fe.reset_mtp_state()
        for p in range(prompt_len):
            fe._static_token_id.copy_(ids[:, p:p + 1])
            g = fe._ensure_graph_for_pos_nvfp4(p)
            fe._replay_pos_graph(g, p)          # ← the wired call
            fe._prefill_h_cache[p:p + 1].copy_(fe._last_hidden_buf.view(1, hidden))

    prefill()                                   # warm (capture + adopt)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record(); prefill(); ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1)


def main():
    fe = Qwen36TorchFrontendRtx(CKPT, device="cuda:0", max_seq=2048, quant="nvfp4")
    ids = fe._tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_len = int(ids.shape[1])

    _ = fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=N, K=K)  # warm
    out = fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=N, K=K)
    new_ids = out[0, prompt_len:].tolist()
    sha = hashlib.sha256(",".join(map(str, new_ids)).encode()).hexdigest()[:16]

    ttft = measure_ttft(fe, ids)
    print(f"RESULT USE_EXEC={USE_EXEC} use_exec_active={getattr(fe, '_use_exec', False)} "
          f"prompt_len={prompt_len} TTFT_ms={ttft:.3f} sha={sha}")


if __name__ == "__main__":
    main()
