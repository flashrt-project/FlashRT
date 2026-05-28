"""Phase-C de-risk — frt_graph adopts the REAL Pi0.5 RTX infer graph.

Pi0.5 (VLA) captures its full infer (vision + encoder + decoder + diffusion)
as ONE framework ctypes CUDAGraph. This proves the exec layer drives that real
non-LLM / non-token / diffusion graph: in one process, with the input buffers
frozen, replay the SAME captured graph two ways and compare the action output
buffer:
  - ctypes CUDAGraph.replay()      (baseline path)
  - frt adopt(_graph_exec) + replay (exec layer, wrapped stream)

forward() is idempotent given fixed inputs, so the two replays must produce a
bit-identical diffusion_noise buffer (cosine 1.0). Avoids the cross-process
RNG nondeterminism that makes a sha gate meaningless for Pi0.5.

Run (inside pi0-stablehlo-test, after building fp16 fa2):
    PYTHONPATH=/workspace/PI/official/FlashRT-spec:/workspace/PI/official/FlashRT-spec/exec/build \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python exec/tests/test_adopt_pi05.py --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch
"""

import argparse
import numpy as np
import torch
import _flashrt_exec as ex

import flash_rt


def _read(buf):
    """Snapshot a CudaBuffer's raw bytes (dtype-agnostic) as uint8 numpy."""
    return buf.download_new((buf.nbytes,), np.uint8).copy()


def _cos(a_u8, b_u8):
    # diffusion_noise is BF16; reinterpret bytes via torch (numpy has no bf16).
    a = torch.frombuffer(a_u8.tobytes(), dtype=torch.bfloat16).float()
    b = torch.frombuffer(b_u8.tobytes(), dtype=torch.bfloat16).float()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--fp8", action="store_true", help="test the FP8 path")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(args.num_views)]

    print(f"precision: {'fp8' if args.fp8 else 'fp16'}")
    model = flash_rt.load_model(
        args.checkpoint, framework="torch", config="pi05", hardware="auto",
        num_views=args.num_views, num_steps=args.steps, cache_frames=1,
        use_fp8=bool(args.fp8), use_fp16=not args.fp8)
    model.predict(images, prompt="pick up the red block")  # builds graph + inputs
    pl = model._pipe.pipeline
    assert getattr(pl, "_graph", None) is not None, "Pi05 full infer graph not captured"
    out_buf = pl.bufs["diffusion_noise"]

    # forward() is IN-PLACE on diffusion_noise (reads initial noise, writes
    # final actions back into the same buffer), so restore a fixed start state
    # before every replay to make the comparison meaningful.
    save = _read(out_buf)

    def restore():
        out_buf.upload(save)
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)

    def replay_ctypes():
        restore()
        pl._graph.replay(pl._graph_stream)
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)
        return _read(out_buf)

    ctx = ex.Ctx()
    gs_id = ctx.wrap_stream(int(pl._graph_stream.value))
    fg = ctx.graph("pi05_infer", 1)
    fg.adopt(0, pl._graph._graph_exec.value)

    def replay_frt():
        restore()
        rc = fg.replay(0, gs_id)
        assert rc == 0, f"frt replay rc={rc}"
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)
        return _read(out_buf)

    a1 = replay_ctypes()
    b = replay_frt()
    a2 = replay_ctypes()   # determinism of the ctypes path from the same start

    ctypes_repro = np.array_equal(a1, a2)
    bit_identical = np.array_equal(a1, b)
    cos = _cos(a1, b)

    print("\n===== ADOPT REAL PI0.5 RTX INFER GRAPH =====")
    print(f"out_buf bytes        : {out_buf.nbytes}")
    print(f"ctypes self-reproduce: {ctypes_repro}")
    print(f"frt == ctypes (exact): {bit_identical}")
    print(f"cosine(frt, ctypes)  : {cos:.6f}")
    assert cos >= 0.999, f"frt-driven replay cosine {cos} below 0.999 red line"
    print("\nPASS — frt adopt+replay matches ctypes on a real Pi0.5 VLA graph "
          f"(cos={cos:.6f}, bit_identical={bit_identical})")


if __name__ == "__main__":
    main()
