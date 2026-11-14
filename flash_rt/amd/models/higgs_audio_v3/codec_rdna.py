"""RDNA execution adapter for the unchanged FP32 Higgs codec model."""
import torch

from flash_rt.amd.hardware.rdna.convolution import dilated_conv1d
from flash_rt.models.higgs_audio_v3.codec import HiggsAudioV3Codec


class _DilatedConv1d(torch.nn.Module):
    """Keep frozen parameters and their state-dict names; change only lowering."""

    def __init__(self, conv):
        super().__init__()
        if (conv.stride != (1,) or conv.padding_mode != "zeros"
                or conv.padding[0] != conv.dilation[0] * (conv.kernel_size[0] // 2)
                or conv.kernel_size[0] % 2 != 1):
            raise ValueError("Higgs RDNA codec requires centered stride-one dilated convolutions")
        self.weight = conv.weight
        self.register_parameter("bias", conv.bias)
        self.dilation, self.padding, self.groups = conv.dilation[0], conv.padding[0], conv.groups

    def forward(self, x):
        return dilated_conv1d(x, self.weight, self.bias, dilation=self.dilation,
                              padding=self.padding, groups=self.groups)


def _prepare_decoder(module):
    for name, child in list(module.named_children()):
        if isinstance(child, _DilatedConv1d):
            continue
        if isinstance(child, torch.nn.Conv1d) and child.dilation[0] > 1:
            setattr(module, name, _DilatedConv1d(child))
        else:
            _prepare_decoder(child)


class HiggsAudioV3CodecRdna(HiggsAudioV3Codec):
    """Use dense library convolutions for the codec's sparse time taps."""

    def __init__(self, model, device):
        super().__init__(model, device)
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError("Higgs RDNA codec requires FP32 weights")
        _prepare_decoder(model.acoustic_decoder)

    @torch.no_grad()
    def decode(self, codes_TN):
        with torch.cuda.device(self.device), torch.autocast("cuda", enabled=False):
            return super().decode(codes_TN)
