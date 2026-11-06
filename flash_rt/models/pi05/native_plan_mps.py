"""Lower the shared Pi0.5 traversal to a bounded native Metal plan."""

import threading
import time

from flash_rt.hardware.apple_native.plan import Attention, Plan, UIntParameter, Weight
from flash_rt.models.pi05.torch_pipeline import PATCHES_PER_VIEW, Pi05TorchPipeline


class NativePi05Plan:
    """Reuse one plan across prompt lengths within the declared capacity."""

    def __init__(
        self,
        source,
        prompt_len,
        *,
        library=None,
        capture_probes=False,
        residency=True,
        split_decode=True,
        merged_encoder_ffn=True,
        int8_options=None,
        dynamic_prompt=True,
        pad_prompt=False,
    ):
        if not source.compact_encoder or not source.precompute_modulation:
            raise ValueError(
                "Native plan requires compact encoder and precomputed modulation"
            )
        if not 0 < prompt_len <= source.max_prompt_len:
            raise ValueError("Invalid initial prompt length")
        if pad_prompt and not dynamic_prompt:
            raise ValueError("pad_prompt requires dynamic_prompt=True")
        if dynamic_prompt and int8_options is not None and int8_options.get("observe"):
            raise ValueError("Calibration requires dynamic_prompt=False to observe only real tokens")
        if int8_options is None:
            self.plan = Plan(library)
        else:
            from flash_rt.hardware.apple_native.int8 import INT8Plan
            self.plan = INT8Plan(library, **int8_options)
        self._lock = threading.Lock()
        self.prompt_len = prompt_len
        self.pad_prompt = pad_prompt
        self.prompt_capacity = source.max_prompt_len if dynamic_prompt else prompt_len
        image_tokens = source.num_views * PATCHES_PER_VIEW
        self._valid_prefix = (
            UIntParameter(self.plan, image_tokens + prompt_len,
                          minimum=image_tokens + 1,
                          maximum=image_tokens + self.prompt_capacity)
            if dynamic_prompt else None
        )
        self.probes = {}
        self.last_timing = None
        try:
            # Reuse the same layer/denoising traversal with build-time descriptors.
            model = object.__new__(Pi05TorchPipeline)
            for key in (
                "num_views",
                "max_prompt_len",
                "prefix_capacity",
                "chunk_size",
                "num_steps",
                "compact_encoder",
                "precompute_modulation",
            ):
                setattr(model, key, getattr(source, key))
            # Host inputs go straight to their mutable compute storage. Separate
            # prompt/noise staging allocations would require replay copy kernels.
            model.buf = {
                k: self.plan.allocate(v.shape)
                for k, v in source.buf.items()
                if k not in ("input_prompt", "input_noise")
            }
            model.w = {k: Weight(v, k) for k, v in source.w.items()}
            model._rope_cos = Weight(source._rope_cos, "rope.cos")
            model._rope_sin = Weight(source._rope_sin, "rope.sin")
            model._positions = range(source.prefix_capacity + source.chunk_size)
            model._decoder_modulations = self.plan.allocate(
                source._decoder_modulations.shape
            )
            self.plan.write(model._decoder_modulations, source._decoder_modulations)
            self.plan.weight_bytes += source._decoder_modulations.numel() * 2
            model.ops = self.plan.ops
            model.ops.merged_encoder_ffn = merged_encoder_ffn
            model.gemm = self.plan
            model.attn = Attention(self.plan, split_decode=split_decode)

            def save(key, value):
                if capture_probes:
                    dst = self.plan.allocate(value.shape)
                    dst.copy_(value)
                    self.probes[key] = dst

            model._save = save
            self.input_images = model.buf["input_images"]
            self.input_noise = self.output = model.buf["noise"]
            self.input_prompt = model.buf["encoder_x"][
                image_tokens:image_tokens + self.prompt_capacity
            ]
            self.plan.stage = "patch"
            patch = self.plan.patch(
                model.buf["vision_x"],
                self.input_images,
                source.w["vision_patch_embedding_w"],
                source.w["vision_patch_embedding_b"],
                source.w["vision_position_embedding"],
            )
            if capture_probes:
                probe = self.plan.allocate((source.num_views, 1152, 16, 16))
                self.plan.dispatch(
                    "tr_patch_probe",
                    [patch, probe],
                    {1},
                    [source.num_views * 256, 1152],
                    ((probe.numel() + 63) // 64,),
                )
                self.probes["patch"] = probe
            self.plan.stage = "vision"
            model._vision_blocks()
            self.plan.stage = "prefix"
            if dynamic_prompt and not pad_prompt:
                valid = self._valid_prefix.dimension()
                extent = valid - image_tokens
                model._encoder(self.input_prompt[:extent], extent, valid_prefix=valid)
            else:
                valid = model._encoder(self.input_prompt, self.prompt_capacity,
                                       valid_prefix=self._valid_prefix)
            self.plan.stage = "decode"
            model._decoder(valid, prefix_storage_len=(
                image_tokens + self.prompt_capacity if pad_prompt else None))
            model._save("final_raw_action", self.output)
            self.plan.finish()
            self.plan.set_resident(residency)
        except BaseException:
            self.plan.close()
            raise

    def close(self):
        with self._lock:
            self.plan.close()

    def _write_inputs(self, images, prompt, noise):
        # Call under _lock: both prompt and noise are overwritten by execution.
        if len(prompt.shape) != 2:
            raise ValueError("Native prompt must be a rank-two embedding array")
        length = prompt.shape[0]
        if not self.accepts_prompt_length(length) or prompt.shape[1:] != self.input_prompt.shape[1:]:
            raise ValueError("Native prompt shape outside the plan's declared capacity")
        self.plan.write(self.input_images, images)
        self.plan.write(self.input_prompt[:length], prompt)
        if self.pad_prompt and length < self.prompt_capacity:
            # Padded queries are computed but never attended to. Restore their
            # finite input each replay, including after a longer prompt.
            self.plan._host_view(self.input_prompt)[length:].fill(0)
        self.plan.write(self.input_noise, noise)
        started = time.perf_counter()
        if self._valid_prefix is not None:
            self._valid_prefix.set(self._valid_prefix.minimum - 1 + length)
        self.last_shape_update_us = (time.perf_counter() - started) * 1e6
        self.prompt_len = length

    def accepts_prompt_length(self, length):
        return (0 < length <= self.prompt_capacity if self._valid_prefix is not None
                else length == self.prompt_len)

    def profile(self, images, prompt, noise):
        """Restore inputs, profile each dispatch and return an owned output."""
        with self._lock:
            if self.plan.handle is None:
                raise RuntimeError("Native plan is closed")
            self._write_inputs(images, prompt, noise)
            timings = self.plan.profile()
            return self.plan.read(self.output), timings

    def forward(self, images, prompt, noise, *, execution="icb", capture_probes=False):
        with self._lock:
            if self.plan.handle is None:
                raise RuntimeError("Native plan is closed")
            if capture_probes and not self.probes:
                raise ValueError("Plan was built without probes")
            start = time.perf_counter()
            self._write_inputs(images, prompt, noise)
            uploaded = time.perf_counter()
            t = self.plan.run(execution)
            executed = time.perf_counter()
            output = self.plan.read(self.output)
            self.last_timing = dict(
                **t,
                shape_update_us=self.last_shape_update_us,
                upload_us=(uploaded - start) * 1e6,
                download_us=(time.perf_counter() - executed) * 1e6,
            )
            probes = (
                {k: self.plan.read(v) for k, v in self.probes.items()}
                if capture_probes
                else {}
            )
            return output, probes
