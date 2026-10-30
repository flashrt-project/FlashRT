"""Bind Apple MPS providers to the shared Pi0.5 BF16 model traversal."""

from contextlib import nullcontext

from flash_rt.hardware.mps.attention import MpsAttentionBackend
from flash_rt.hardware.torch_reference import TorchGemmBackend, TorchTensorOps
from flash_rt.models.pi05.torch_pipeline import Pi05TorchPipeline


class Pi05PipelineMps(Pi05TorchPipeline):
    def __init__(self, weights, *, num_views=2, max_prompt_len=200,
                 chunk_size=10, num_steps=10):
        super().__init__(
            weights, TorchTensorOps(), TorchGemmBackend(), MpsAttentionBackend(),
            num_views=num_views, max_prompt_len=max_prompt_len,
            chunk_size=chunk_size, num_steps=num_steps, device="mps",
        )
        self._native_pipeline = None
        self._stage_profiler = None

    def enable_native_pipeline(self, prompt_len, *, library=None, capture_probes=False, residency=True,
                               split_decode=True, merged_encoder_ffn=True):
        """Build one ICB for patch + blocks + projector + prefix + denoising.

        Shape specialization occurs here, outside replay. Use forward_native
        for CPU inputs and CPU actions without MPS input/output staging.
        """
        from flash_rt.models.pi05.native_plan_mps import NativePi05Plan
        native = NativePi05Plan(self, prompt_len, library=library,
                               capture_probes=capture_probes, residency=residency,
                               split_decode=split_decode,
                               merged_encoder_ffn=merged_encoder_ffn)
        previous = self._native_pipeline
        self._native_pipeline = native
        if previous is not None:
            previous.close()
        self.invalidate_encoder_cache()

    def forward_native(self, images_nhwc, prompt_embeds, noise, *, capture_probes=False):
        """Host input -> shared Metal buffers -> owned FP32 CPU action array."""
        if self._native_pipeline is None:
            raise RuntimeError("enable_native_pipeline must be called first")
        self.invalidate_encoder_cache()
        with self._range("native.full_pipeline"):
            output, probes = self._native_pipeline.forward(
                images_nhwc, prompt_embeds, noise.reshape(self.chunk_size, 32),
                capture_probes=capture_probes)
        if capture_probes:
            import torch
            self.probes = {key: torch.from_numpy(value) for key, value in probes.items()}
        return output

    def disable_native_pipeline(self):
        if self._native_pipeline is not None:
            self._native_pipeline.close()
            self._native_pipeline = None
        self.invalidate_encoder_cache()

    def forward_with_inputs(self, images_nhwc, prompt_embeds, prompt_len, noise,
                            *, capture_probes=False, use_graph=False):
        if self._native_pipeline is None:
            return super().forward_with_inputs(images_nhwc, prompt_embeds, prompt_len,
                noise, capture_probes=capture_probes, use_graph=use_graph)
        import torch
        self.invalidate_encoder_cache()
        self._validate_inputs(images_nhwc, prompt_embeds, prompt_len, noise)
        if use_graph:
            raise ValueError("Use native ICB replay directly, not CUDA graph mode")
        if prompt_len != self._native_pipeline.prompt_len:
            raise ValueError("Native plan prompt length changed; call enable_native_pipeline with the new length")
        if capture_probes and not self._native_pipeline.probes:
            raise ValueError("Build the native plan with capture_probes=True for diagnostic output")
        self._capture_probes = capture_probes
        if capture_probes:
            self.probes = {}
        try:
            output = self.forward_native(images_nhwc, prompt_embeds, noise,
                                         capture_probes=capture_probes)
            with self._range("native.action_to_mps"):
                self.buf["noise"].copy_(torch.from_numpy(output))
            self._save("final_raw_action", self.buf["noise"])
            # Native K/V belongs to the native plan, not the torch decode-only cache.
            return self.buf["noise"].unsqueeze(0)
        finally:
            self._capture_probes = False

    def _vision_blocks(self):
        with self._range("vision.blocks"):
            return super()._vision_blocks()

    def profile(self, **kwargs):
        """Opt into Instruments ranges; see profile_pipeline for options."""
        from flash_rt.hardware.apple_native.profiling import profile_pipeline
        return profile_pipeline(self, **kwargs)

    def _range(self, name):
        if self._stage_profiler is None:
            return nullcontext()
        return self._stage_profiler.range(name)

    def _vision(self, images_nhwc):
        with self._range("vision.patch_and_blocks"):
            return super()._vision(images_nhwc)

    def _vision_patch(self, images_nhwc):
        with self._range("vision.patch"):
            return super()._vision_patch(images_nhwc)

    def _vision_block(self, index):
        with self._range(f"vision.layer.{index}"):
            return super()._vision_block(index)

    def _vision_project(self):
        with self._range("vision.project"):
            return super()._vision_project()

    def _encoder(self, *args, **kwargs):
        with self._range("prefix.with_vision_projection"):
            return super()._encoder(*args, **kwargs)

    def _decoder(self, *args, **kwargs):
        with self._range("action.denoise"):
            return super()._decoder(*args, **kwargs)
