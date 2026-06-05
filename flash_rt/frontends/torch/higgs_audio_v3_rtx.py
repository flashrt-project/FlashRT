"""Higgs Audio v3 TTS-4B — RTX SM120 PyTorch frontend (BF16 acoustic path).

Hand-written forward over flash_rt_kernels + flash_rt_fa2: a dense Qwen3-4B
backbone drives a fused multi-codebook head that emits 8 codec tokens per
acoustic frame under a delay pattern, decoded autoregressively with greedy
sampling. ``predict`` returns raw ``[T, num_codebooks]`` codes; waveform
synthesis is the codec's responsibility.

Kernels used: ``rms_norm``, ``bf16_matmul_qwen36_bf16``, ``silu_mul_qwen36_bf16``,
``qwen3_q_norm_rope_qstage_bf16`` / ``qwen3_k_norm_rope_kvwrite_bf16`` (fused
q/k-norm + full RoPE), and vendored FlashAttention-2 via
``RtxFlashAttnBackendQwen3`` (head config 36 / 32q / 8kv / 128 is shared with
Qwen3-8B). Plain RMSNorm weight convention (multiplies by ``w``, not ``1+w``).
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F

from flash_rt.models.higgs_audio_v3.pipeline_rtx import HiggsAudioV3Dims

_SPECIALS = ("<|tts|>", "<|text|>", "<|audio|>")


class HiggsAudioV3TorchFrontendRtx:
    """Single-stream Higgs Audio v3 TTS inference on RTX SM120 (BF16 backbone)."""

    DIMS = HiggsAudioV3Dims()

    def __init__(self, checkpoint_path: str, *,
                 device: str = "cuda:0",
                 max_seq: int = 2048,
                 max_new_frames: int = 1024,
                 max_prefill_tokens: int = 512,
                 fp8: bool = True,
                 alloc_own_forward_buffers: bool = True) -> None:
        """Load the BF16 backbone + fused codebook table and build scratch.

        Args:
          checkpoint_path: dir holding config.json + model.safetensors.
          device: CUDA device string.
          max_seq: KV-cache length (prompt + generated frames).
          max_new_frames: cap on generated acoustic frames per call.
          max_prefill_tokens: max prompt length served by the batched single-
            pass FP8 prefill; longer prompts fall back to the eager per-token
            prefill. Sizes the attention Q/O buffers.
          fp8: run the decode backbone in FP8 W8A8 (the dedicated M=1 GEMV
            path); falls back to the BF16 backbone when False. FP8 weights are
            quantised and activation scales calibrated lazily on first use.
          alloc_own_forward_buffers: pre-allocate the attention backend and
            RoPE table at construction.
        """
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)
        self.max_new_frames = int(max_new_frames)
        self.max_prefill_tokens = int(max_prefill_tokens)
        self.fp8 = bool(fp8)
        # Batched single-pass prompt prefill (M=P) instead of P eager M=1 steps.
        # On by default for FP8; set FLASHRT_HIGGS_BATCHED_PREFILL=0 for eager.
        self._batched_prefill = self.fp8 and os.environ.get(
            "FLASHRT_HIGGS_BATCHED_PREFILL", "1") != "0"
        # Single position-agnostic decode CUDA graph (devpos KV-write + FA2
        # seqused_k). On by default for FP8; set FLASHRT_HIGGS_GRAPH=0 to force
        # the eager decode path.
        self._use_graph = self.fp8 and os.environ.get(
            "FLASHRT_HIGGS_GRAPH", "1") != "0"

        self._tokenizer: Any = None
        self._special_ids: dict[str, int] = {}
        self._weights: dict[str, Any] | None = None
        self._cfg: dict[str, Any] | None = None
        self._attn = None
        self._rope_cos = None
        self._rope_sin = None
        self._prompt_ids: list[int] | None = None
        self._resident_ids: list[int] | None = None   # KV currently on GPU
        self._fp8_decoder: Any = None
        self._codec: Any = None
        self.latency_records: list[float] = []

        self._load_weights()
        if alloc_own_forward_buffers:
            self._alloc_buffers()
            self._build_rope_table()

    # ── Load ──

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

    # ── Buffers / RoPE ──

    def _alloc_buffers(self) -> None:
        from flash_rt.hardware.rtx.attn_backend_qwen3 import (
            RtxFlashAttnBackendQwen3,
        )

        # max_q_seq sizes the Q/O/lse buffers; batched prefill needs room for
        # the whole prompt (q_seq=P), decode only ever uses q_seq=1.
        mqs = max(1, self.max_prefill_tokens) if self._batched_prefill else 1
        self._attn = RtxFlashAttnBackendQwen3(
            max_seq=self.max_seq, max_q_seq=mqs, dtype=torch.bfloat16)
        nc = self._cfg["num_codebooks"]
        self._cb_offsets = (
            torch.arange(nc, device=self.device) * self._cfg["codebook_vocab"])

    def _build_rope_table(self) -> None:
        hd = self._cfg["head_dim"]
        theta = self._cfg["rope_theta"]
        inv = 1.0 / (theta ** (
            torch.arange(0, hd, 2, device=self.device, dtype=torch.float32) / hd))
        pos = torch.arange(self.max_seq, device=self.device, dtype=torch.float32)
        f = torch.outer(pos, inv)  # [max_seq, hd/2]
        self._rope_cos = f.cos().to(torch.bfloat16).contiguous()
        self._rope_sin = f.sin().to(torch.bfloat16).contiguous()

    # ── Forward ──

    def _layer(self, fvk, L, h, t):
        c = self._cfg
        H, NQ, NKV, HD = (c["hidden"], c["num_q_heads"],
                          c["num_kv_heads"], c["head_dim"])
        INTER, eps = c["intermediate"], c["rms_norm_eps"]
        w = self._weights["layers"][L]
        s = torch.cuda.current_stream().cuda_stream
        bf16 = torch.bfloat16

        def rms(x, weight, M, K):
            out = torch.empty(M, K, device=self.device, dtype=bf16)
            fvk.rms_norm(x.contiguous().data_ptr(), weight.data_ptr(),
                         out.data_ptr(), M, K, eps, s)
            return out

        def mm(x, weight, M, N, K):
            out = torch.empty(M, N, device=self.device, dtype=bf16)
            fvk.bf16_matmul_qwen36_bf16(x.contiguous().data_ptr(),
                                        weight.data_ptr(), out.data_ptr(),
                                        M, N, K, s)
            return out

        xn = rms(h, w["in_norm"], 1, H)
        q = mm(xn, w["q"], 1, NQ * HD, H).view(NQ, HD).contiguous()
        k = mm(xn, w["k"], 1, NKV * HD, H).view(NKV, HD).contiguous()
        v = mm(xn, w["v"], 1, NKV * HD, H).view(NKV, HD).contiguous()
        cos_t = self._rope_cos[t]
        sin_t = self._rope_sin[t]
        fvk.qwen3_q_norm_rope_qstage_bf16(
            q_pre=q.data_ptr(), q_norm_w=w["qn"].data_ptr(),
            cos=cos_t.data_ptr(), sin=sin_t.data_ptr(),
            q_buf_dst=self._attn.Q_buf[:, :1].data_ptr(),
            n_q_heads=NQ, eps=eps, stream=s)
        fvk.qwen3_k_norm_rope_kvwrite_bf16(
            k_pre=k.data_ptr(), v_pre=v.data_ptr(), k_norm_w=w["kn"].data_ptr(),
            cos=cos_t.data_ptr(), sin=sin_t.data_ptr(),
            k_cache_dst=self._attn.K_cache[L, t:t + 1].data_ptr(),
            v_cache_dst=self._attn.V_cache[L, t:t + 1].data_ptr(),
            n_kv_heads=NKV, eps=eps, stream=s)
        self._attn.run("full", L, 1, kv_seq=t + 1, causal=True,
                       softmax_scale=HD ** -0.5)
        ao = self._attn.O_buf[:, :1].reshape(1, NQ * HD).to(bf16)
        h = (h.float() + mm(ao, w["o"], 1, H, NQ * HD).float()).to(bf16)
        xn2 = rms(h, w["post_norm"], 1, H)
        gate = mm(xn2, w["gate"], 1, INTER, H)
        up = mm(xn2, w["up"], 1, INTER, H)
        act = torch.empty(1, INTER, device=self.device, dtype=bf16)
        fvk.silu_mul_qwen36_bf16(gate.contiguous().data_ptr(),
                                 up.contiguous().data_ptr(),
                                 act.data_ptr(), INTER, s)
        h = (h.float() + mm(act, w["down"], 1, H, INTER).float()).to(bf16)
        return h

    def _step(self, fvk, embed_row, t):
        c = self._cfg
        H = c["hidden"]
        h = embed_row
        for L in range(c["num_layers"]):
            h = self._layer(fvk, L, h, t)
        s = torch.cuda.current_stream().cuda_stream
        hn = torch.empty(1, H, device=self.device, dtype=torch.bfloat16)
        fvk.rms_norm(h.contiguous().data_ptr(),
                     self._weights["final_norm"].data_ptr(),
                     hn.data_ptr(), 1, H, c["rms_norm_eps"], s)
        return hn

    def _logits(self, hn):
        cb = self._weights["codebook"]
        nc, cv = self._cfg["num_codebooks"], self._cfg["codebook_vocab"]
        return (hn.float() @ cb.float().t()).view(nc, cv)

    def _embed_codes(self, codes):
        cb = self._weights["codebook"]
        ids = codes.to(self.device).long() + self._cb_offsets
        return F.embedding(ids, cb).sum(0, keepdim=True)

    # ── Public API ──

    def build_prompt(self, text: str, system: str | None = None) -> list[int]:
        """Zero-shot TTS prompt: <|tts|> <|text|> [enc(system)] tok(text) <|audio|>.

        An optional ``system`` preamble (voice/style instruction) is prepended
        as a clean reusable token prefix: requests that share the same ``system``
        share that leading run, so its KV can be reused across requests (serving
        prefix caching) — only the varying ``text`` suffix is re-prefilled.
        """
        ids = [self._special_ids["<|tts|>"], self._special_ids["<|text|>"]]
        if system:
            ids += self._tokenizer.encode(system, add_special_tokens=False)
        ids += self._tokenizer.encode(text, add_special_tokens=False)
        ids.append(self._special_ids["<|audio|>"])
        return ids

    def set_prompt(self, text: str) -> None:
        self._prompt_ids = self.build_prompt(text)
        self._attn.reset_cache()
        self._resident_ids = None      # KV zeroed; nothing reusable resident

    # ── FP8 / codec lazy init ──

    def _ensure_fp8(self) -> None:
        if self._fp8_decoder is not None:
            return
        from flash_rt.frontends.torch._higgs_audio_v3_fp8 import (
            HiggsAudioV3Fp8Decoder,
        )
        dec = HiggsAudioV3Fp8Decoder(self)
        dec.calibrate(self._prompt_ids)   # static activation scales (one-time)
        self._fp8_decoder = dec

    def _ensure_codec(self) -> None:
        if self._codec is not None:
            return
        from flash_rt.models.higgs_audio_v3.codec import HiggsAudioV3Codec
        self._codec = HiggsAudioV3Codec.from_checkpoint(
            self.checkpoint_path, device=self.device)

    def _frame_logits(self, fvk, embed_row, t):
        """[num_codebooks, codebook_vocab] logits for one frame at position t."""
        if self.fp8:
            self._fp8_decoder.set_input(embed_row)
            return self._fp8_decoder.step(t)[0]
        return self._logits(self._step(fvk, embed_row, t))

    def _decode_logits(self, fvk, embed_row, t):
        """Decode-position logits; uses the position-agnostic graph when on."""
        if self._use_graph:
            dec = self._fp8_decoder
            if getattr(dec, "_graph", None) is None:
                dec.capture_graph(embed_row, t)   # one-time, any position
            return dec.decode_graph(embed_row, t)[0]
        return self._frame_logits(fvk, embed_row, t)

    # ── split surface (committed streaming seam for the serving layer) ──
    # prefill once -> decode_stream yields un-delayed [nc] frames as they
    # complete. The delay pattern means an un-delayed frame t is only complete
    # after nc-1 further frames decode, so the stream carries a fixed nc-1
    # holdback; every yielded frame is already committed to the KV state.

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

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int] | None = None,
                cached_tokens: int = 0) -> int:
        """Run the prompt prefill; returns cur_pos (= prompt length).

        ``cached_tokens`` > 0 reuses the KV of the leading ``cached_tokens``
        tokens (already resident from a prior prefill of the same prefix) and
        prefills only the suffix — serving prefix caching. The claim is verified
        against the resident tokens; on mismatch it safely falls back to a full
        prefill. Reuse requires the batched FP8 prefill path.
        """
        from flash_rt import flash_rt_kernels as fvk
        if prompt_ids is not None:
            self._prompt_ids = list(prompt_ids)
        if self._prompt_ids is None:
            raise RuntimeError("no prompt set; call set_prompt()/pass prompt_ids")
        if self.fp8:
            self._ensure_fp8()
        P = len(self._prompt_ids)
        batched = self._batched_prefill and P <= self._attn._max_q_seq
        # Validate the cached-prefix claim against the resident KV (safety: never
        # reuse stale/mismatched KV); reuse only on the batched path.
        reuse = (batched and 0 < cached_tokens < P
                 and self._resident_ids is not None
                 and len(self._resident_ids) >= cached_tokens
                 and self._prompt_ids[:cached_tokens]
                 == self._resident_ids[:cached_tokens])
        if not reuse:
            self._attn.reset_cache()      # cold prefill writes KV from pos 0
            cached_tokens = 0
        if batched:
            self._gen_logits = self._fp8_decoder.prefill_batched(
                self._prompt_ids, start_pos=cached_tokens)[0]
        else:
            te = self._weights["text_embed"]
            for t, tok in enumerate(self._prompt_ids):
                row = F.embedding(torch.tensor([tok], device=self.device), te)
                self._gen_logits = self._frame_logits(fvk, row, t)
        self._gen_pos = P
        self._resident_ids = list(self._prompt_ids)
        return self._gen_pos

    @torch.no_grad()
    def decode_stream(self):
        """Yield committed un-delayed ``[nc]`` int code frames (cpu) as ready."""
        from flash_rt import flash_rt_kernels as fvk
        nc = self._cfg["num_codebooks"]
        boc, eoc = self.DIMS.boc_id, self.DIMS.eoc_id
        P, logits = self._gen_pos, self._gen_logits
        delay, eoc_countdown, done = 0, None, False
        window: list[torch.Tensor] = []
        for j in range(self.max_new_frames):
            codes = logits.argmax(-1).clone()
            if delay < nc:
                if delay + 1 < nc:
                    codes[delay + 1:] = boc
                delay += 1
            elif eoc_countdown is not None:
                eoc_countdown -= 1
                if eoc_countdown <= 0:
                    done = True
            elif int(codes[0]) == eoc:
                eoc_countdown = nc - 2
            if done:
                break
            window.append(codes.clone())
            if len(window) >= nc:                       # frame (len-nc) complete
                base = len(window) - nc
                yield torch.stack(
                    [window[base + i][i] for i in range(nc)]).cpu()
            logits = self._decode_logits(fvk, self._embed_codes(codes), P + j)

    @torch.no_grad()
    def predict(self, text: str | None = None) -> torch.Tensor:
        """Generate acoustic codes for ``text`` (greedy, delay/EOC).

        Returns raw codes of shape ``[T, num_codebooks]`` (int64, CPU);
        feed them to :meth:`synthesize` (or the Higgs codec) for a 24 kHz wave.
        """
        import time

        if text is not None:
            self.set_prompt(text)
        nc = self._cfg["num_codebooks"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.prefill()
        frames = list(self.decode_stream())
        torch.cuda.synchronize()
        self.latency_records.append((time.perf_counter() - t0) * 1000.0)
        if not frames:
            return torch.empty(0, nc, dtype=torch.long)
        return torch.stack(frames)             # [T, nc] un-delayed

    @torch.no_grad()
    def synthesize(self, codes: torch.Tensor) -> torch.Tensor:
        """``[T, num_codebooks]`` codes -> mono 24 kHz waveform ``[L]`` (cpu)."""
        self._ensure_codec()
        if codes.numel() == 0:
            return torch.zeros(0)
        return self._codec.decode(codes)

    @torch.no_grad()
    def generate(self, text: str) -> torch.Tensor:
        """Full pipeline: text -> acoustic codes -> 24 kHz waveform ``[L]``."""
        return self.synthesize(self.predict(text))

    SAMPLES_PER_FRAME = 960   # 24000 Hz / 25 Hz acoustic frame rate

    @torch.no_grad()
    def generate_stream(self, text: str, *, system: str | None = None,
                        first_chunk: int = 8, chunk: int = 25, ctx: int = 8,
                        holdback: int = 8):
        """Stream 24 kHz audio chunks as frames decode (low TTFA).

        Yields mono waveform chunks (cpu f32). ``first_chunk`` frames are emitted
        as soon as they are committed (minimises time-to-first-audio), then
        ``chunk`` frames at a time. The codec conv has a receptive field, so each
        emitted frame is decoded inside a window with ``ctx`` frames of already-
        emitted left context and ``holdback`` frames of not-yet-emitted right
        context; only the centre frames' samples are released, so the streamed
        waveform matches the one-shot ``synthesize`` (no boundary seams).

        ``system`` is an optional shared preamble (voice/style instruction). When
        successive requests carry the same ``system``, its KV is reused across
        requests (only the new ``text`` suffix is prefilled) — serving prefix
        caching, bit-identical to a cold prefill.
        """
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
        for frame in self.decode_stream():
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
