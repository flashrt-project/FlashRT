"""P8 validation: AWQ in-place requant skips graph recapture.

Loads Pi05TorchFrontendThorFP4 with use_awq=True across all 17 live layers, runs the
AWQ recalibration path (which now updates packed/sfb buffers in place),
then compares cos vs the pytorch fp32 reference and the FP8 prod canary.

Gates:
  - cos vs /tmp/v_torch.npy >= 0.997  (current AWQ baseline is 0.9979)
  - cos vs /tmp/v_prod.npy  >= 0.996  (current AWQ baseline is 0.9965)
  - no exception thrown by the in-place path

Run in container:
    PYTHONPATH=<repo_root> \
      python3 tests/test_p8_awq_inplace.py
"""
import sys
import time
import numpy as np
import torch


def cos(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    np.random.seed(42)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": wrist}

    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4

    print("Building Pi0.5 AWQ full-18 (use_awq=True, P8 in-place requant)...")
    pipe = Pi05TorchFrontendThorFP4(
        os.environ.get("FLASH_RT_PI05_CKPT", "/workspace/pytorch_checkpoints/pi05_libero_converted"),
        num_views=2, autotune=3,
        use_fp4_encoder_ffn=True,
        fp4_layers=tuple(range(17)),
        use_awq=True,
        awq_alpha=0.5,
    )

    t0 = time.perf_counter()
    pipe.set_prompt("pick up the red block and place it in the tray")
    print(f"set_prompt (cold): {(time.perf_counter()-t0)*1000:.1f}ms")

    t0 = time.perf_counter()
    for _ in range(5):
        pipe.infer(obs)
    print(f"5 warmup infer: {(time.perf_counter()-t0)*1000:.1f}ms")

    matched_noise = torch.from_numpy(np.load("/tmp/matched_noise.npy")).cuda()
    _orig = torch.Tensor.normal_

    def _patched(self, *a, **kw):
        if self.data_ptr() == pipe._g_noise.data_ptr():
            self.copy_(matched_noise); return self
        return _orig(self, *a, **kw)
    torch.Tensor.normal_ = _patched
    r = pipe.infer(obs)
    torch.Tensor.normal_ = _orig
    actions = r["actions"]

    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        pipe.infer(obs)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p25, p50, p75 = lat[5], lat[10], lat[15]

    v_torch = np.load("/tmp/v_torch.npy")
    v_prod = np.load("/tmp/v_prod.npy")
    cos_torch = cos(actions, v_torch)
    cos_prod = cos(actions, v_prod)

    print()
    print(f"  cos vs pytorch fp32 ref : {cos_torch:.6f}")
    print(f"  cos vs FP8 prod canary  : {cos_prod:.6f}")
    print(f"  P50 latency             : {p50:.2f} ms  (p25={p25:.2f} p75={p75:.2f})")
    print()
    print("Gates:")
    g1 = cos_torch >= 0.997
    g2 = cos_prod >= 0.996
    print(f"  cos vs pytorch >= 0.997 : {'PASS' if g1 else 'FAIL'} ({cos_torch:.6f})")
    print(f"  cos vs prod    >= 0.996 : {'PASS' if g2 else 'FAIL'} ({cos_prod:.6f})")
    return 0 if (g1 and g2) else 1


if __name__ == "__main__":
    sys.exit(main())
