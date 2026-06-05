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
                 fp8: bool = True,
                 alloc_own_forward_buffers: bool = True) -> None:
        """Load the BF16 backbone + fused codebook table and build scratch.

        Args:
          checkpoint_path: dir holding config.json + model.safetensors.
          device: CUDA device string.
          max_seq: KV-cache length (prompt + generated frames).
          max_new_frames: cap on generated acoustic frames per call.
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
        self.fp8 = bool(fp8)

        self._tokenizer: Any = None
        self._special_ids: dict[str, int] = {}
        self._weights: dict[str, Any] | None = None
        self._cfg: dict[str, Any] | None = None
        self._attn = None
        self._rope_cos = None
        self._rope_sin = None
        self._prompt_ids: list[int] | None = None
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

        self._attn = RtxFlashAttnBackendQwen3(
            max_seq=self.max_seq, max_q_seq=1, dtype=torch.bfloat16)
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

    def build_prompt(self, text: str) -> list[int]:
        """Zero-shot TTS prompt: <|tts|> <|text|> tok(text) <|audio|>."""
        ids = [self._special_ids["<|tts|>"], self._special_ids["<|text|>"]]
        ids += self._tokenizer.encode(text, add_special_tokens=False)
        ids.append(self._special_ids["<|audio|>"])
        return ids

    def set_prompt(self, text: str) -> None:
        self._prompt_ids = self.build_prompt(text)
        self._attn.reset_cache()

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

    @torch.no_grad()
    def predict(self, text: str | None = None) -> torch.Tensor:
        """Generate acoustic codes for ``text`` (greedy, delay/EOC).

        Returns raw codes of shape ``[T, num_codebooks]`` (int64, CPU);
        feed them to :meth:`synthesize` (or the Higgs codec) for a 24 kHz wave.
        """
        import time

        from flash_rt import flash_rt_kernels as fvk

        if text is not None:
            self.set_prompt(text)
        if self._prompt_ids is None:
            raise RuntimeError("no prompt set; call set_prompt() or pass text")
        if self.fp8:
            self._ensure_fp8()

        c = self._cfg
        nc = c["num_codebooks"]
        boc, eoc = self.DIMS.boc_id, self.DIMS.eoc_id
        te = self._weights["text_embed"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._attn.reset_cache()
        ids = self._prompt_ids
        for t, tok in enumerate(ids):
            row = F.embedding(torch.tensor([tok], device=self.device), te)
            logits = self._frame_logits(fvk, row, t)
        P = len(ids)

        delay, eoc_countdown, done = 0, None, False
        delayed = []
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
            delayed.append(codes.clone())
            logits = self._frame_logits(fvk, self._embed_codes(codes), P + j)

        torch.cuda.synchronize()
        self.latency_records.append((time.perf_counter() - t0) * 1000.0)
        if not delayed:
            return torch.empty(0, nc, dtype=torch.long)
        grid = torch.stack(delayed)            # [L, nc] delayed
        T = grid.shape[0] - (nc - 1)
        if T <= 0:
            return torch.empty(0, nc, dtype=torch.long)
        return torch.stack([grid[i:i + T, i] for i in range(nc)], dim=1).cpu()

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
