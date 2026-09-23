"""Portable tensor providers for reference execution of shared torch plans.

These are explicit reference providers, never an automatic production fallback.
They do not load a vendor extension and work on CPU or a torch GPU device.
"""
import torch
import torch.nn.functional as F


class TorchTensorOps:
    fused_rope = False
    merged_encoder_ffn = False
    merged_decoder_ffn = False

    def layer_norm(self, out, x, weight, bias, eps=1e-6):
        out.copy_(F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps))

    def rms_norm(self, out, x, weight=None, eps=1e-6):
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            value = value * (1 + weight.float())
        out.copy_(value)

    def adarms(self, out, x, cond, weight, bias, eps=1e-6, modulation=None):
        if cond is not None:
            torch.addmm(bias, cond, weight, out=modulation)
        scale, shift, gate = modulation.chunk(3, -1)
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        out.copy_(value * (1 + scale.float()) + shift.float())
        return gate

    def gelu(self, out, x):
        out.copy_(F.gelu(x.float(), approximate="tanh"))

    def gelu_mul(self, out, gate, up):
        out.copy_(F.gelu(gate.float(), approximate="tanh") * up.float())

    def silu(self, out, x):
        out.copy_(F.silu(x.float()))

    def residual(self, out, update, residual, gate=None):
        value = update.float()
        if gate is not None:
            value = value * gate.float()
        out.copy_(residual.float() + value)

    def residual_rms(self, out_sum, out_norm, update, residual, eps=1e-6):
        self.residual(out_sum, update, residual)
        self.rms_norm(out_norm, out_sum, eps=eps)

    def residual_adarms(self, out_sum, out_norm, update, residual, gate,
                        cond, weight, bias, eps=1e-6, modulation=None):
        self.residual(out_sum, update, residual, gate)
        return self.adarms(out_norm, out_sum, cond, weight, bias, eps, modulation)


class TorchGemmBackend:
    def linear(self, out, x, weight, bias=None):
        value = x @ weight
        # Match separate BF16 bias addition of the unfused provider.
        if bias is not None:
            value = value + bias
        out.copy_(value)

    def linear_residual(self, out, x, weight, bias=None):
        return False


class TorchAttentionBackend:
    def __init__(self, vision_heads=16):
        self.vision_heads = vision_heads

    def vision(self, qkv, out=None):
        views, seq, channels = qkv.shape
        q, k, v = qkv.reshape(views, seq, 3, self.vision_heads, channels // (3 * self.vision_heads)).unbind(2)
        value = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        value = value.transpose(1, 2).reshape(views, seq, channels // 3)
        if out is None:
            return value
        out.copy_(value)
        return out

    def gqa(self, q, k, v, *, valid_prefix, prefix_capacity, out=None):
        mask = None
        if k.shape[0] not in (valid_prefix, valid_prefix + q.shape[0]):
            mask = torch.zeros(q.shape[0], k.shape[0], dtype=torch.bool, device=q.device)
            mask[:, :valid_prefix] = True
            mask[:, prefix_capacity:] = True
        value = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), attn_mask=mask, enable_gqa=True)
        value = value.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1)
        if out is None:
            return value
        out.copy_(value)
        return out
