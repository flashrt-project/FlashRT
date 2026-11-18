"""Native HIP BF16 routed experts with persistent RDNA workspace."""

import torch


class RdnaMoe:
    """Keep [expert, out, in] weights and FP32 down results until slot summation.

    The activated, weighted intermediate rounds to BF16. The returned FP32
    buffer is overwritten by the next call; all workspace is bound before capture.
    """

    def __init__(
        self,
        tokens,
        hidden,
        experts,
        top_k,
        intermediate,
        *,
        device,
        dtype=torch.bfloat16,
        kernels=None,
    ):
        if dtype != torch.bfloat16:
            raise ValueError("RDNA MoE requires BF16 inputs and weights")
        if kernels is None:
            from flash_rt.amd import flash_rt_amd_kernels as kernels
        self.kernels = kernels
        self._packed_weights = {}
        self._compressed_weights = {}
        self.tokens, self.hidden = tokens, hidden
        self.experts, self.top_k, self.intermediate = experts, top_k, intermediate
        # Top-k selects distinct experts, so each receives at most one row per token.
        self.counts = torch.empty(experts, dtype=torch.int32, device=device)
        self.rows = torch.empty((experts, tokens), dtype=torch.int32, device=device)
        self.slots = torch.empty_like(self.rows)
        self.inter = torch.empty((tokens, top_k, intermediate), dtype=dtype, device=device)
        self.routes_out = torch.empty((tokens, top_k, hidden), dtype=torch.float32, device=device)
        self.out = torch.empty((tokens, hidden), dtype=torch.float32, device=device)

    def prepare_weights(self, gate, up, down, *, compress=False):
        """Pack immutable expert weights once, before graph capture."""
        arch = dict(self.kernels.build_info())["gpu_arch"]
        if arch != "gfx1151" or self.hidden % 32 or self.intermediate % 32:
            return

        def pack(weight):
            experts, columns, inner = weight.shape
            return weight.view(experts, columns // 16, 16, inner // 16, 16).permute(0, 1, 3, 2, 4)

        gate_up = torch.stack((pack(gate), pack(up)), dim=-2).contiguous()
        packed_down = pack(down).contiguous()
        if compress:
            gate_up, gate_metadata = self._pack_compact(gate_up, gate=True)
            packed_down, down_metadata = self._pack_compact(packed_down, gate=False)
            self._compressed_weights[gate.data_ptr()] = gate_metadata, down_metadata
        else:
            self._compressed_weights.pop(gate.data_ptr(), None)
        # Retain the original tensors so their addresses remain valid cache keys.
        self._packed_weights[gate.data_ptr()] = (gate, up, down, gate_up, packed_down)

    def _pack_compact(self, raw, *, gate):
        # Keep encoded and raw fallback tiles in one contiguous buffer.
        elements = 512 if gate else 256
        raw = raw.view(-1, elements)
        high = (raw.view(torch.int16) >> 8) & 0x78
        low, upper = high.amin(1), high.amax(1)
        headers = torch.where((low == upper) & (low != 0), low, 0).int()
        lengths = torch.where(headers != 0, elements * 3 // 256, elements // 64).int()
        offsets = torch.cumsum(lengths, 0, dtype=torch.int32) - lengths
        # Low 24 bits address 128-byte units; the high byte stores shared exponent bits.
        metadata = offsets | (headers << 24)
        blob = torch.empty(int(lengths.sum()) * 128, dtype=torch.uint8, device=raw.device)
        self.kernels.moe_pack_compact_rdna(
            raw.data_ptr(), blob.data_ptr(), metadata.data_ptr(), raw.shape[0], gate,
            int(torch.cuda.current_stream(raw.device).cuda_stream),
        )
        return blob, metadata

    def __call__(
        self, hidden_states, routing_weights, selected_experts, gate_weight, up_weight, down_weight
    ):
        selected = selected_experts.to(torch.int32).contiguous()
        route = routing_weights.contiguous()
        prepared = self._packed_weights.get(gate_weight.data_ptr())
        if prepared is None:
            forward = self.kernels.moe_forward_rdna
            weights = gate_weight.data_ptr(), up_weight.data_ptr(), down_weight.data_ptr()
        else:
            forward = self.kernels.moe_forward_packed_rdna
            weights = prepared[3].data_ptr(), prepared[4].data_ptr()
            compressed = self._compressed_weights.get(gate_weight.data_ptr())
            if compressed is not None:
                forward = self.kernels.moe_forward_compact_rdna
                weights += tuple(t.data_ptr() for t in compressed)
        forward(
            hidden_states.data_ptr(), route.data_ptr(), selected.data_ptr(), *weights,
            self.counts.data_ptr(), self.rows.data_ptr(), self.slots.data_ptr(),
            self.inter.data_ptr(), self.routes_out.data_ptr(), self.out.data_ptr(), self.tokens,
            self.hidden, self.experts, self.top_k, self.intermediate,
            int(torch.cuda.current_stream(hidden_states.device).cuda_stream),
        )
        return self.out
