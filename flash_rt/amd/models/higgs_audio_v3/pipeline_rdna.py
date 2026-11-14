"""One Higgs Qwen3 semantic traversal for RDNA BF16 and FP8 execution.

The frontend owns weights, cache, position and scratch. Target-selected linear
operations lower the four projections without architecture forks in this loop.
"""
from flash_rt.amd.hardware.rdna.cached_attention import cached_attention
from flash_rt.amd.hardware.rdna.transformer import (
    qk_norm_rope_cache, rms_norm, silu_mul,
)


class HiggsAudioV3PipelineRdna:
    def __init__(self, weights, config, linear, k_cache, v_cache, position, cos, sin):
        self.weights, self.cfg, self.linear = weights, config, linear
        self.k_cache, self.v_cache, self.position = k_cache, v_cache, position
        self.cos, self.sin = cos, sin

    def forward(self, buffers, *, weights=None, linear=None):
        """Run one preallocated batch at the current device position."""
        w, c, b = self.weights if weights is None else weights, self.cfg, buffers
        linear = self.linear if linear is None else linear
        h = b["h"]
        rms_norm(h, w["layers"][0]["in_norm"], b["norm"], eps=c["rms_norm_eps"])
        for index, layer in enumerate(w["layers"]):
            linear(b["norm"], layer["qkv"], b["qkv"], quant_scratch=b["quant_h"])
            qk_norm_rope_cache(
                b["qkv"], layer["qn"], layer["kn"], self.cos, self.sin,
                self.position, b["q"], self.k_cache[index], self.v_cache[index],
                eps=c["rms_norm_eps"])
            cached_attention(b["q"], self.k_cache[index], self.v_cache[index],
                             self.position, b["attention"], workspace=b["attention_workspace"])
            linear(b["attention"], layer["o"], b["update"], quant_scratch=b["quant_q"])
            rms_norm(h, layer["post_norm"], b["norm"], eps=c["rms_norm_eps"],
                     residual=b["update"])
            linear(b["norm"], layer["gu"], b["gate_up"], quant_scratch=b["quant_h"])
            silu_mul(b["gate_up"], b["activation"])
            linear(b["activation"], layer["down"], b["update"], quant_scratch=b["quant_i"])
            # Carry the down projection into the following norm so its FP32
            # residual sum is not rounded before computing the statistics.
            next_norm = (w["layers"][index + 1]["in_norm"]
                         if index + 1 < len(w["layers"]) else w["final_norm"])
            rms_norm(h, next_norm, b["norm"], eps=c["rms_norm_eps"], residual=b["update"])
        linear(b["norm"], w["head"], b["logits"])
        return b["logits"]
