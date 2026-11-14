"""Higgs Audio v3 BF16/FP8 frontend for gfx1100, gfx1151 and gfx1201."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
import warnings

import torch

from flash_rt.amd.hardware.capabilities_rdna import device_capabilities
from flash_rt.models.higgs_audio_v3 import HiggsAudioV3Dims


@dataclass(frozen=True)
class _HiggsRdnaTarget:
    native_fp8: bool
    fp8_long_k_rows: int = 1

    def select_fp8(self, requested, gpu_arch):
        if requested and not self.native_fp8:
            warnings.warn(
                f"Higgs Audio v3 FP8 W8A8 is not supported on {gpu_arch}; "
                "falling back to BF16 W16A16.", RuntimeWarning, stacklevel=3)
        return self.native_fp8 and requested is not False

    def make_linear(self, fp8):
        from flash_rt.amd.hardware.rdna.linear import RdnaLinear
        return RdnaLinear(fp8_activations=fp8, fp8_long_k_rows=self.fp8_long_k_rows)

    def precision_spec(self, fp8):
        return {
            "weights": "float8_e4m3fn" if fp8 else "bfloat16",
            "decode": "fp8_w8a8_dynamic" if fp8 else "bf16_w16a16",
            "prefill": "fp8_w8a8_dynamic" if fp8 else "bf16_w16a16",
            "single_token_prefill": "fp8_w8a8_dynamic" if fp8 else "bf16_w16a16",
            "scales": "per_tensor_float32" if fp8 else None,
            "activation_scales": "dynamic_per_row_float32" if fp8 else None,
            "kv_cache": "bfloat16",
            "codec": "float32",
            "audio_head": "bfloat16",
            "logits": "bfloat16",
            "embeddings": "bfloat16",
            "accumulator": "float32",
        }


_TARGETS = {
    # RDNA4 adds native E4M3 WMMA for the FP8 prefill. Older targets use BF16;
    # compressed W8A16 execution is not the model's public FP8 mode.
    "gfx1100": _HiggsRdnaTarget(native_fp8=False),
    "gfx1151": _HiggsRdnaTarget(native_fp8=False),
    "gfx1201": _HiggsRdnaTarget(native_fp8=True, fp8_long_k_rows=4),
}


def _pack_weight(weight, *, fp8):
    """Pack Higgs per-tensor E4M3 weights with CUDA-equivalent scaling."""
    from flash_rt.amd.hardware.rdna.linear import LinearWeight, pack_weight as backend_pack_weight

    # Reuse the backend's layout/finite checks and BF16 representation.
    packed = backend_pack_weight(weight, fp8=False)
    if not fp8:
        return packed
    values = weight.float()
    # Match CUDA's host-scalar division as well as its scale value: device-
    # tensor division can round differently at E4M3 midpoint boundaries.
    scale = (values.abs().amax() / 448).item()
    if scale == 0:
        scale = 1.0
    quantized = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    # The shared RDNA kernels take an N-element scale vector. Repeating the
    # scalar preserves their ABI without changing the per-tensor definition.
    return LinearWeight(quantized.view(torch.uint8).contiguous(),
                        torch.full((weight.shape[0],), scale, device=weight.device,
                                   dtype=torch.float32))


_SPECIALS = ("<|tts|>", "<|text|>", "<|audio|>")
_GREEDY_TEMP_THRESHOLD = 1e-5
_MAX_SAMPLING_SEED = (1 << 64) - 1
_REPEAT_STOP_FRAMES = 32


def _delayed_eoc_countdown(codes: torch.Tensor, nc: int, eoc: int) -> int | None:
    """Remaining delayed rows to flush once any codebook emits EOC."""
    eoc_pos = (codes == eoc).nonzero(as_tuple=False)
    if not eoc_pos.numel():
        return None
    first_eoc_cb = int(eoc_pos[0])
    return max(1, nc - 1 - first_eoc_cb)


def _repeat_code_run(codes: torch.Tensor,
                     prev_key: tuple[int, ...] | None,
                     count: int) -> tuple[tuple[int, ...], int]:
    key = tuple(int(x) for x in codes.tolist())
    return key, count + 1 if key == prev_key else 1


def _normalize_sampling_options(
    temperature: float, seed: int | None
) -> tuple[float, int]:
    """Validate public sampling options and materialize one request seed."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be a finite value >= 0")
    if seed is None:
        seed = (int.from_bytes(os.urandom(8), "little")
                if temperature > _GREEDY_TEMP_THRESHOLD else 0)
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if not 0 <= seed <= _MAX_SAMPLING_SEED:
        raise ValueError("seed must be in [0, 2**64 - 1]")
    return temperature, seed


class HiggsAudioV3TorchFrontendAmdRdna:
    """The CUDA Higgs API with explicit RDNA precision and cache ownership.

    ``fp8=None`` selects W8A8 on gfx1201 and BF16 on gfx1100/gfx1151.
    Explicit FP8 on a target without native FP8 warns and falls back to BF16.
    W8A8 uses per-tensor OCP E4M3 weights and dynamic per-row activations.
    Codec and KV remain FP32/BF16.
    Inspect ``precision_spec`` for the actual execution tiers.
    """

    DIMS = HiggsAudioV3Dims()
    SAMPLES_PER_FRAME = 960
    last_finish_reason: str | None = None

    def __init__(self, checkpoint_path: str, *, device: str = "cuda:0",
                 max_seq: int = 2048, max_new_frames: int = 1024,
                 max_prefill_tokens: int = 512, fp8: bool | None = None,
                 alloc_own_forward_buffers: bool = True, hardware=None):
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("Higgs Audio v3 RDNA requires PyTorch with ROCm")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Higgs Audio v3 RDNA requires a ROCm device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if fp8 is not None and not isinstance(fp8, bool):
            raise TypeError("fp8 must be a bool or None")
        for name, value in (("max_seq", max_seq), ("max_new_frames", max_new_frames),
                            ("max_prefill_tokens", max_prefill_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_seq > self.DIMS.max_pos:
            raise ValueError(f"max_seq exceeds supported position capacity {self.DIMS.max_pos}")
        self.max_seq, self.max_new_frames = max_seq, max_new_frames
        self.max_prefill_tokens = min(max_prefill_tokens, max_seq)
        self.checkpoint_path = str(checkpoint_path)
        self._batched_prefill = os.getenv("FLASHRT_HIGGS_BATCHED_PREFILL", "1") != "0"
        self._use_graph = os.getenv("FLASHRT_HIGGS_GRAPH", "1") != "0"
        self._tokenizer = self._weights = self._cfg = self._codec = None
        self._prompt_ids = self._resident_ids = None
        self.pipeline = self._graph = self._sample_codes = None
        self._rope_cos = self._rope_sin = None
        self._decode_buffers = None
        self._prefill_buffers = None
        self._prefill_storage = None
        self._prefill_weights = None
        self.latency_records = []
        with torch.cuda.device(self.device):
            props = torch.cuda.get_device_properties(self.device)
            self._caps = device_capabilities(props.gcnArchName, hardware)
            self.hardware = self._caps.hardware
            self._target = _TARGETS[self._caps.gpu_arch]
            self.fp8 = self._target.select_fp8(fp8, self._caps.gpu_arch)
            self.precision_spec = self._target.precision_spec(self.fp8)
            self.precision_spec["swiglu"] = "bfloat16_silu_then_multiply"
            self.precision_spec["attention_denominator"] = "float32_probabilities"
            if not self._batched_prefill or self.max_prefill_tokens == 1:
                self.precision_spec["prefill"] = self.precision_spec["single_token_prefill"]
            self._linear = self._target.make_linear(self.fp8)
            self._prefill_linear = self._linear
            self._load_weights()
            if alloc_own_forward_buffers:
                self._alloc_buffers()
                self._build_rope_table()
        logging.getLogger(__name__).info(
            "Higgs Audio v3 on %s: %s", self._caps.gpu_arch, self.precision_spec)

    def _load_weights(self) -> None:
        from safetensors.torch import load_file

        cfg = json.load(open(os.path.join(self.checkpoint_path, "config.json")))
        tc = cfg["text_config"]
        enc = cfg["audio_encoder_config"]
        rope_theta = float(tc.get("rope_parameters", {}).get(
            "rope_theta", tc.get("rope_theta", self.DIMS.rope_theta)))
        self._cfg = {
            "hidden": int(tc["hidden_size"]),
            "num_layers": int(tc["num_hidden_layers"]),
            "num_q_heads": int(tc["num_attention_heads"]),
            "num_kv_heads": int(tc["num_key_value_heads"]),
            "head_dim": int(tc["head_dim"]),
            "intermediate": int(tc["intermediate_size"]),
            "rms_norm_eps": float(tc["rms_norm_eps"]),
            "rope_theta": rope_theta,
            "num_codebooks": int(enc["num_codebooks"]),
            "codebook_vocab": int(enc["vocab_size"]),
        }
        self._assert_dims()

        dev, bf16 = self.device, torch.bfloat16
        sd = load_file(os.path.join(self.checkpoint_path, "model.safetensors"))

        def g(k):
            return sd[k].to(dev, bf16)

        nl = self._cfg["num_layers"]
        layers = []
        for i in range(nl):
            p = f"body.layers.{i}"
            layers.append({
                "in_norm": g(f"{p}.input_layernorm.weight"),
                "q": g(f"{p}.self_attn.q_proj.weight"),
                "k": g(f"{p}.self_attn.k_proj.weight"),
                "v": g(f"{p}.self_attn.v_proj.weight"),
                "o": g(f"{p}.self_attn.o_proj.weight"),
                "qn": g(f"{p}.self_attn.q_norm.weight"),
                "kn": g(f"{p}.self_attn.k_norm.weight"),
                "post_norm": g(f"{p}.post_attention_layernorm.weight"),
                "gate": g(f"{p}.mlp.gate_proj.weight"),
                "up": g(f"{p}.mlp.up_proj.weight"),
                "down": g(f"{p}.mlp.down_proj.weight"),
            })
        self._weights = {
            "layers": layers,
            "text_embed": g("tied.embedding.text_embedding.weight"),
            "final_norm": g("body.norm.weight"),
            # fused [num_codebooks * codebook_vocab, hidden]; head ties to embed.
            "codebook": g(
                "tied.embedding.modality_embeddings.0.embedding.weight"),
        }
        self._load_tokenizer()

    def _assert_dims(self) -> None:
        d, c = self.DIMS, self._cfg
        for name, want, got in (
            ("hidden", d.hidden, c["hidden"]),
            ("num_layers", d.num_layers, c["num_layers"]),
            ("num_q_heads", d.num_q_heads, c["num_q_heads"]),
            ("num_kv_heads", d.num_kv_heads, c["num_kv_heads"]),
            ("head_dim", d.head_dim, c["head_dim"]),
            ("intermediate", d.intermediate, c["intermediate"]),
            ("num_codebooks", d.num_codebooks, c["num_codebooks"]),
            ("codebook_vocab", d.codebook_vocab, c["codebook_vocab"]),
        ):
            if want != got:
                raise ValueError(
                    f"checkpoint dim {name}={got} != expected {want}; "
                    f"this frontend targets Higgs Audio v3 TTS-4B.")

    def _load_tokenizer(self) -> None:
        # Load tokenizer.json directly: transformers<5 mishandles the
        # list-form extra_special_tokens in this checkpoint's config.
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        raw = Tokenizer.from_file(
            os.path.join(self.checkpoint_path, "tokenizer.json"))
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
        vocab = dict(self._tokenizer.get_added_vocab())
        missing = [t for t in _SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"tokenizer missing Higgs TTS specials: {missing}")
        self._special_ids = {t: vocab[t] for t in _SPECIALS}

    def _alloc_buffers(self):
        if self.pipeline is not None:
            raise RuntimeError("cannot replace buffers of an initialized Higgs decoder")
        c = self._cfg
        shape = (c["num_layers"], self.max_seq, c["num_kv_heads"], c["head_dim"])
        self._k_cache = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        self._v_cache = torch.empty_like(self._k_cache)
        self._position = torch.zeros((), dtype=torch.int32, device=self.device)
        self._codes_dev = torch.empty(c["num_codebooks"], dtype=torch.long, device=self.device)
        self._embed_buf = torch.empty((1, c["hidden"]), dtype=torch.bfloat16, device=self.device)
        self._decode_buffers = self._allocate_batch(1)

    def _build_rope_table(self) -> None:
        hd = self._cfg["head_dim"]
        theta = self._cfg["rope_theta"]
        inv = 1.0 / (theta ** (
            torch.arange(0, hd, 2, device=self.device, dtype=torch.float32) / hd))
        pos = torch.arange(self.max_seq, device=self.device, dtype=torch.float32)
        f = torch.outer(pos, inv)  # [max_seq, hd/2]
        self._rope_cos = f.cos().to(torch.bfloat16).contiguous()
        self._rope_sin = f.sin().to(torch.bfloat16).contiguous()

    def _allocate_batch(self, rows, *, linear=None):
        from flash_rt.amd.hardware.rdna.cached_attention import allocate_decode_workspace

        c = self._cfg
        linear = self._linear if linear is None else linear
        h, q, kv, inter = (c["hidden"], c["num_q_heads"] * c["head_dim"],
                            c["num_kv_heads"] * c["head_dim"], c["intermediate"])
        def tensor(width):
            return torch.empty((rows, width), device=self.device, dtype=torch.bfloat16)
        def quant(width):
            if not linear.fp8_activations:
                return None
            return (torch.empty((rows, width), device=self.device, dtype=torch.uint8),
                    torch.empty(rows, device=self.device, dtype=torch.float32))
        return {
            "h": tensor(h), "norm": tensor(h), "qkv": tensor(q + 2 * kv),
            "q": tensor(q).view(rows, c["num_q_heads"], c["head_dim"]),
            "attention": tensor(q), "update": tensor(h), "gate_up": tensor(2 * inter),
            "activation": tensor(inter),
            # Match CUDA native's BF16 head output before greedy/CDF sampling.
            "logits": tensor(c["num_codebooks"] * c["codebook_vocab"]),
            "quant_h": quant(h), "quant_q": quant(q), "quant_i": quant(inter),
            "attention_workspace": allocate_decode_workspace(c["num_q_heads"], c["head_dim"], self.device)
                                   if rows == 1 else None,
        }

    def _prefill_batch(self, rows):
        # Single-token prefill uses the same M=1 buffers and attention workspace
        # as decode. Their addresses remain fixed for the lifetime of the graph.
        if rows == 1:
            return self._decode_buffers
        if self._prefill_storage is None or self._prefill_storage["h"].shape[0] < rows:
            self._prefill_storage = self._allocate_batch(rows, linear=self._prefill_linear)
            self._prefill_buffers = None
        if self._prefill_buffers is None or self._prefill_buffers["h"].shape[0] != rows:
            self._prefill_buffers = {}
            for name, value in self._prefill_storage.items():
                if value is None or name == "attention_workspace":
                    view = value
                elif isinstance(value, tuple):
                    view = tuple(item[:rows] for item in value)
                else:
                    view = value[:rows]
                self._prefill_buffers[name] = view
        return self._prefill_buffers

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return
        if self._decode_buffers is None or self._rope_cos is None or self._rope_sin is None:
            raise RuntimeError("allocate Higgs RDNA buffers and RoPE tables before inference")
        from flash_rt.amd.models.higgs_audio_v3.pipeline_rdna import HiggsAudioV3PipelineRdna
        from flash_rt.amd.models.higgs_audio_v3.sampling import delayed_codebook_sample_embed_bf16

        with torch.cuda.device(self.device):
            for layer in self._weights["layers"]:
                qkv = torch.cat([layer.pop(k) for k in ("q", "k", "v")], 0).contiguous()
                gu = torch.cat([layer.pop(k) for k in ("gate", "up")], 0).contiguous()
                layer["qkv"] = _pack_weight(qkv, fp8=self.fp8)
                layer["gu"] = _pack_weight(gu, fp8=self.fp8)
                layer["o"] = _pack_weight(layer["o"], fp8=self.fp8)
                layer["down"] = _pack_weight(layer["down"], fp8=self.fp8)
            del qkv, gu
            # Reuse the BF16 tied embedding for the head. Its precision is
            # independent of the backbone's FP8 weight packing.
            self._weights["head"] = _pack_weight(self._weights["codebook"], fp8=False)
            self.pipeline = HiggsAudioV3PipelineRdna(
                self._weights, self._cfg, self._linear, self._k_cache, self._v_cache,
                self._position, self._rope_cos, self._rope_sin)
            self._sample_codes = delayed_codebook_sample_embed_bf16
            torch.cuda.empty_cache()

    def build_prompt(self, text: str, system: str | None = None) -> list[int]:
        """Zero-shot TTS prompt: <|tts|> <|text|> [enc(system)] tok(text) <|audio|>.

        An optional ``system`` preamble (voice/style instruction) is prepended
        as a clean reusable token prefix: requests that share the same ``system``
        share that leading run, so its KV can be reused across requests (serving
        prefix caching) — only the varying ``text`` suffix is re-prefilled.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        if system is not None and not isinstance(system, str):
            raise TypeError("system must be a string or None")
        ids = [self._special_ids["<|tts|>"], self._special_ids["<|text|>"]]
        if system:
            ids += self._tokenizer.encode(system, add_special_tokens=False)
        ids += self._tokenizer.encode(text, add_special_tokens=False)
        ids.append(self._special_ids["<|audio|>"])
        return ids

    def set_prompt(self, text: str):
        self._prompt_ids = self.build_prompt(text)
        self._resident_ids = None

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int] | None = None, cached_tokens: int = 0) -> int:
        if prompt_ids is not None:
            self._prompt_ids = list(prompt_ids)
        if self._prompt_ids is None:
            raise RuntimeError("no prompt set; call set_prompt()/pass prompt_ids")
        ids = self._prompt_ids
        if (not ids or any(not isinstance(t, int) or isinstance(t, bool)
                           or not 0 <= t < self._weights["text_embed"].shape[0] for t in ids)):
            raise ValueError("prompt_ids must contain valid integer text-token IDs")
        if len(ids) >= self.max_seq:
            raise ValueError("prompt must leave at least one position for audio decoding")
        if not isinstance(cached_tokens, int) or isinstance(cached_tokens, bool) or cached_tokens < 0:
            raise ValueError("cached_tokens must be a nonnegative integer")
        reuse = (0 < cached_tokens < len(ids) and self._resident_ids is not None
                 and self._resident_ids[:cached_tokens] == ids[:cached_tokens]
                 and len(self._resident_ids) >= cached_tokens and self._batched_prefill)
        start = cached_tokens if reuse else 0
        self._resident_ids = None
        with torch.cuda.device(self.device):
            self._ensure_pipeline()
            chunk_size = self.max_prefill_tokens if self._batched_prefill else 1
            while start < len(ids):
                stop = min(start + chunk_size, len(ids))
                rows = stop - start
                b = self._prefill_batch(rows)
                tokens = torch.tensor(ids[start:stop], device=self.device, dtype=torch.long)
                torch.index_select(self._weights["text_embed"], 0, tokens, out=b["h"])
                self._position.fill_(start)
                output = self.pipeline.forward(b, weights=self._prefill_weights,
                                               linear=self._prefill_linear)
                start = stop
            self._gen_logits = output[-1].view(self._cfg["num_codebooks"], self._cfg["codebook_vocab"])
            self._gen_pos = len(ids)
            self._resident_ids = list(ids)
        return self._gen_pos

    def resident_prefix_len(self, prompt_ids: list[int]) -> int:
        """Longest prefix of ``prompt_ids`` whose KV is already resident on GPU
        (from the previous prefill). The serving layer passes this back as
        ``cached_tokens`` to skip re-prefilling a shared prefix (system/voice
        preamble) — prefix-KV reuse. Mechanism only; the cache/eviction policy
        lives in the serving layer. Always keeps >=1 token to prefill fresh."""
        resident = self._resident_ids
        if not resident:
            return 0
        n = 0
        for a, b in zip(prompt_ids, resident):
            if a != b:
                break
            n += 1
        return max(0, min(n, len(prompt_ids) - 1))

    def _sample_logits(self, logits, delay, temperature, seed, step):
        with torch.cuda.device(self.device):
            self._sample_codes(
                logits, self._weights["codebook"], self._codes_dev, self._embed_buf,
                delay=delay, boc=self.DIMS.boc_id, temperature=temperature, seed=seed, step=step)

    def _capture_graph(self, embed_row, position):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._position.fill_(position)
            for _ in range(3):
                self._decode_buffers["h"].copy_(embed_row)
                self.pipeline.forward(self._decode_buffers)
        stream.synchronize()
        with torch.cuda.stream(stream):
            self._decode_buffers["h"].copy_(embed_row)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.pipeline.forward(self._decode_buffers)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._graph = graph

    def _decode_logits(self, embed_row, t):
        if t >= self.max_seq:
            raise ValueError("Higgs Audio v3 generation exhausted max_seq KV capacity")
        with torch.cuda.device(self.device):
            if self._use_graph and self._graph is None:
                self._capture_graph(embed_row, t)
            self._decode_buffers["h"].copy_(embed_row)
            self._position.fill_(t)
            if self._use_graph:
                self._graph.replay()
                output = self._decode_buffers["logits"]
            else:
                output = self.pipeline.forward(self._decode_buffers)
        return output.view(self._cfg["num_codebooks"], self._cfg["codebook_vocab"])

    @torch.no_grad()
    def decode_stream(self, *, temperature: float = 0.0,
                      seed: int | None = None):
        """Yield committed un-delayed ``[nc]`` code frames as ready.

        Greedy decode (``temperature=0``, the default) retains the original
        fused argmax/embed kernel. A positive temperature selects the explicit
        backend's full-vocabulary categorical sampler. A fixed ``seed``
        fixes the request's random stream.
        """
        if getattr(self, "_gen_logits", None) is None:
            raise RuntimeError("call prefill() before decode_stream()")
        nc = self._cfg["num_codebooks"]
        eoc = self.DIMS.eoc_id
        P, logits = self._gen_pos, self._gen_logits
        temperature, sampling_seed = _normalize_sampling_options(
            temperature, seed)
        delay, eoc_countdown, done = 0, None, False
        repeat_key, repeat_count = None, 0
        window: list[torch.Tensor] = []
        capacity = getattr(self, "max_seq", P + self.max_new_frames) - P
        steps = min(self.max_new_frames, max(0, capacity))
        self.last_finish_reason = "max_seq" if steps < self.max_new_frames else "max_new_frames"
        for j in range(steps):
            self._sample_logits(logits, delay, temperature, sampling_seed, j)
            codes = self._codes_dev.cpu()
            if delay < nc:
                delay += 1
            elif eoc_countdown is not None:
                eoc_countdown -= 1
                if eoc_countdown <= 0:
                    done = True
                    self.last_finish_reason = "eoc"
            else:
                eoc_countdown = _delayed_eoc_countdown(codes, nc, eoc)
                if eoc_countdown is None:
                    repeat_key, repeat_count = _repeat_code_run(
                        codes, repeat_key, repeat_count)
                    if repeat_count >= _REPEAT_STOP_FRAMES:
                        done = True
                        self.last_finish_reason = "repeated_code"
            if done:
                break
            window.append(codes.clone())
            if len(window) >= nc:                       # frame (len-nc) complete
                base = len(window) - nc
                frame = torch.stack([window[base + i][i] for i in range(nc)]).cpu()
                yield frame
            logits = self._decode_logits(self._embed_buf, P + j)

    @torch.no_grad()
    def predict(self, text: str | None = None, *, temperature: float = 0.0,
                seed: int | None = None) -> torch.Tensor:
        """Generate acoustic codes for ``text`` (delay/EOC).

        Returns raw codes of shape ``[T, num_codebooks]`` (int64, CPU);
        feed them to :meth:`synthesize` (or the Higgs codec) for a 24 kHz wave.
        The default is the fully fused greedy path. A positive ``temperature``
        samples the full codebook vocabulary.
        """
        import time

        with torch.cuda.device(self.device):
            temperature, seed = _normalize_sampling_options(temperature, seed)
            if text is not None:
                self.set_prompt(text)
            nc = self._cfg["num_codebooks"]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.prefill()
            frames = list(self.decode_stream(temperature=temperature, seed=seed))
            torch.cuda.synchronize()
            self.latency_records.append((time.perf_counter() - t0) * 1000.0)
            if not frames:
                return torch.empty(0, nc, dtype=torch.long)
            return torch.stack(frames)             # [T, nc] un-delayed

    def _ensure_codec(self) -> None:
        if self._codec is not None:
            return
        from flash_rt.amd.models.higgs_audio_v3.codec_rdna import HiggsAudioV3CodecRdna
        self._codec = HiggsAudioV3CodecRdna.from_checkpoint(
            self.checkpoint_path, device=self.device)

    @torch.no_grad()
    def synthesize(self, codes: torch.Tensor) -> torch.Tensor:
        """``[T, num_codebooks]`` codes -> mono 24 kHz waveform ``[L]`` (cpu)."""
        self._ensure_codec()
        if codes.numel() == 0:
            return torch.zeros(0, dtype=torch.float32)
        return self._codec.decode(codes)

    @torch.no_grad()
    def generate(self, text: str, *, temperature: float = 0.0,
                 seed: int | None = None) -> torch.Tensor:
        """Full pipeline: text -> acoustic codes -> 24 kHz waveform ``[L]``."""
        return self.synthesize(self.predict(
            text, temperature=temperature, seed=seed))

    @torch.no_grad()
    def generate_stream(self, text: str, *, system: str | None = None,
                        first_chunk: int = 8, chunk: int = 25, ctx: int = 8,
                        holdback: int = 8, temperature: float = 0.0,
                        seed: int | None = None):
        """Stream 24 kHz audio chunks as frames decode (low TTFA).

        Yields mono waveform chunks (cpu f32). ``first_chunk`` frames are emitted
        as soon as they are committed (minimises time-to-first-audio), then
        ``chunk`` frames at a time. The codec conv has a receptive field, so each
        emitted frame is decoded inside a window with ``ctx`` frames of already-
        emitted left context and ``holdback`` frames of not-yet-emitted right
        context; only the centre frames' samples are released. The default
        eight-frame contexts approximate one-shot ``synthesize``.

        ``system`` is an optional shared preamble (voice/style instruction). When
        successive requests carry the same ``system``, its KV is reused across
        requests (only the new ``text`` suffix is prefilled) — serving prefix
        caching. Hardware-specific docs state the numerical reuse guarantee.
        """
        for name, value, minimum in (("first_chunk", first_chunk, 1), ("chunk", chunk, 1),
                                     ("ctx", ctx, 0), ("holdback", holdback, 0)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        temperature, seed = _normalize_sampling_options(temperature, seed)
        self._ensure_codec()
        if system is None:
            self.set_prompt(text)
            self.prefill()
        else:
            ids = self.build_prompt(text, system=system)
            # reuse the resident shared-preamble KV; only prefill the new suffix
            self.prefill(ids, cached_tokens=self.resident_prefix_len(ids))
        spf = self.SAMPLES_PER_FRAME
        frames: list[torch.Tensor] = []
        emitted = 0

        def flush(ready: int):
            nonlocal emitted
            if ready <= emitted:
                return None
            left = min(ctx, emitted)
            right = min(holdback, len(frames) - ready)
            wav = self._codec.decode(torch.stack(frames[emitted - left:ready + right]))
            n = ready - emitted
            out = wav[left * spf:(left + n) * spf].clone()
            emitted = ready
            return out

        target = first_chunk
        for frame in self.decode_stream(temperature=temperature, seed=seed):
            frames.append(frame)
            ready = len(frames) - holdback           # frames with full right ctx
            if ready - emitted >= target:
                out = flush(ready)
                if out is not None:
                    yield out
                    target = chunk
        out = flush(len(frames))                     # tail (no right context)
        if out is not None:
            yield out
