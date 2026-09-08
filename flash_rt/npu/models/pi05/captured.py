"""Pi0.5 NPU graph construction; tensor work runs only during setup/capture."""
from typing import Optional
import torch
import torch.nn.functional as F
from flash_rt.npu.models.pi05 import pipeline as npu_pl
from flash_rt.npu.models.pi05 import fast as npu_fast


class _CapturedRunner:
    """One captured full-frame execution at a fixed prompt length.

    All device buffers are allocated once; between frames only the
    image/noise contents change (``copy_`` into the same storage), which
    is exactly what the captured graph replays.
    """

    def __init__(self, wb, num_views: int, lang_len: int, chunk: int,
                 num_steps: int, conds, wfast=None, styles=None, wfe=None,
                 norm_stats=None, decoder_rope=None):
        self.num_steps = num_steps
        self.decoder_rope = decoder_rope
        self.chunk = chunk
        self.num_views = num_views
        self.lang_len = lang_len
        self.wb = wb
        self.conds = conds
        self.wfast = wfast            # merged-GEMM decoder overlay (or None)
        self.styles = styles          # precomputed AdaRMS styles (or None)
        self.wfe = wfe                # merged-GEMM encoder overlay (or None)
        if wfast is not None and styles is not None:
            plen = (num_views * npu_pl.VIS_TOKENS_PER_VIEW + lang_len)
            self.kbufs, self.vbufs = npu_fast.make_kv_buffers(plen, chunk)
            self.cos_t, self.sin_t = npu_fast.make_rope_tables(
                plen + chunk, npu_pl.DEC_HD, device="npu")
        else:
            self.kbufs = self.vbufs = None
            self.cos_t = self.sin_t = None
        # one persistent zero row, expanded per call inside the fused LN
        self.z = torch.zeros(1, npu_pl.VIS_D, dtype=torch.bfloat16,
                             device="npu")
        self.imgs = torch.empty(num_views, 3, npu_pl.IMG_HW, npu_pl.IMG_HW,
                                dtype=torch.bfloat16, device="npu")
        self.lang = torch.empty(lang_len, npu_pl.ENC_D, dtype=torch.bfloat16,
                                device="npu")
        self.noise = torch.empty(chunk, npu_pl.ACTION_DIM, dtype=torch.float32,
                                 device="npu")
        self.out = torch.empty(chunk, npu_pl.ACTION_DIM, dtype=torch.float32,
                               device="npu")
        self._graph = None
        self.native = None
        self.norm_stats = norm_stats
        if norm_stats is not None:
            import numpy as np
            from flash_rt.npu.core.acl_runtime import AclRuntime, PinnedBuffer
            self.runtime = AclRuntime()
            self.stream = torch.npu.Stream()
            self.raw_images = torch.zeros(num_views, npu_pl.IMG_HW, npu_pl.IMG_HW, 3,
                                          dtype=torch.uint8, device="npu")
            self.host_images = PinnedBuffer(self.runtime, tuple(self.raw_images.shape), np.uint8)
            self.host_noise = PinnedBuffer(self.runtime, (chunk, npu_pl.ACTION_DIM), np.float32)
            self.host_raw = PinnedBuffer(self.runtime, (chunk, npu_pl.ACTION_DIM), np.float32)
            self.host_robot = PinnedBuffer(self.runtime, (chunk, 7), np.float32)
            self.robot = torch.empty(chunk, 7, device="npu", dtype=torch.float32)
            low = torch.tensor(norm_stats["actions"]["q01"][:7], device="npu", dtype=torch.float32)
            high = torch.tensor(norm_stats["actions"]["q99"][:7], device="npu", dtype=torch.float32)
            self.action_low = low
            self.action_range = high - low + 1e-6

    # -- static prefix length used by the captured graph -----------------
    @property
    def prefix_len(self) -> int:
        return self.num_views * npu_pl.VIS_TOKENS_PER_VIEW + self.lang_len

    def _run(self):
        if self.norm_stats is not None:
            self.imgs.copy_((self.raw_images.permute(0, 3, 1, 2).float() / 127.5 - 1.0)
                            .to(torch.bfloat16))
        vis = npu_fast.vision_tower_opt(self.imgs, self.wb, self.z)
        pref = torch.cat([vis, self.lang], dim=0)
        if self.wfe is not None:
            cache = npu_fast.encoder_pass_opt(pref, self.wfe,
                                              self.cos_t, self.sin_t)
        else:
            cache = npu_pl.encoder_pass(pref, self.wb)
        x_t = self.noise
        plen = pref.shape[0]
        fast = self.wfast is not None and self.styles is not None
        if fast:
            npu_fast.fill_kv_prefix(cache, self.kbufs, self.vbufs, plen)
        for s in range(self.num_steps):
            act = F.linear(x_t, self.wb["action_in_proj.weight"],
                           self.wb["action_in_proj.bias"])
            if fast:
                attn, mlp, top = self.styles
                out = npu_fast.decoder_step_fast_nocat(
                    act, self.kbufs, self.vbufs, self.wfast, attn[s], mlp[s],
                    top[s], plen, self.chunk, self.cos_t, self.sin_t,
                    rope_kernel=self.decoder_rope)
            else:
                out = npu_pl._decoder_step(act, cache, self.conds[s], self.wb,
                                           plen, self.chunk)
            v = F.linear(out, self.wb["action_out_proj.weight"],
                         self.wb["action_out_proj.bias"])
            x_t = x_t - (1.0 / self.num_steps) * v
        self.out.copy_(x_t)
        if self.norm_stats is not None:
            self.robot.copy_((self.out[:, :7].clamp(-1.0, 1.0) + 1.0) / 2.0
                             * self.action_range + self.action_low)

    def capture(self):
        from flash_rt.npu.core.npu_graph import NpuGraph
        # warm-up replays so aclnn workspace/layouts are stable before capture
        for _ in range(3):
            self._run()
        torch.npu.synchronize()
        g = NpuGraph(stream=getattr(self, "stream", None))
        with g:
            self._run()
            if self.norm_stats is not None:
                handle = self.runtime.capture_handle(self.stream.npu_stream)
        torch.npu.synchronize()
        self._graph = g
        if self.norm_stats is not None:
            from flash_rt.npu.core.acl_runtime import NativeReplay
            # Snapshot every external tensor dependency, including weights
            # and setup constants. The borrower can outlive this runner.
            def tensors(value):
                if isinstance(value, torch.Tensor):
                    yield value
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from tensors(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        yield from tensors(item)
                elif hasattr(value, "__dataclass_fields__"):
                    for name in value.__dataclass_fields__:
                        yield from tensors(getattr(value, name))
            dependencies = tuple(tensors(tuple(self.__dict__.values())))
            self.native = NativeReplay(
                self.runtime, handle, self.stream.npu_stream,
                owner=(g, self.stream, dependencies,
                       tuple(v for k, v in self.__dict__.items() if k != "native")),
                inputs=[(self.host_images, self.raw_images.data_ptr()),
                        (self.host_noise, self.noise.data_ptr())],
                outputs=[(self.out.data_ptr(), self.host_raw),
                         (self.robot.data_ptr(), self.host_robot)])

    def replay(self):
        self._graph.replay()

    def fill(self, imgs_norm: torch.Tensor, noise: Optional[torch.Tensor]):
        """imgs_norm (num_views,3,224,224) on npu; noise (chunk,32) npu or None."""
        self.imgs.copy_(imgs_norm)
        if noise is None:
            self.noise.normal_()
        else:
            self.noise.copy_(noise)
        torch.npu.synchronize()
