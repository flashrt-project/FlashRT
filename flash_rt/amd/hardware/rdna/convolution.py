"""FP32 dilated convolution through independent dense polyphase batches."""
import torch.nn.functional as F


def dilated_conv1d(x, weight, bias=None, *, dilation, padding, groups=1):
    """Lower a centered, stride-one convolution without inserting input zeros.

    Splitting the time axis by its residue modulo ``dilation`` converts the
    operation to a dense convolution over a larger batch. The output is
    interleaved back into the original time order, including non-divisible
    lengths. This avoids ROCm's slow generic dilated FP32 convolution path.
    """
    if (x.ndim != 3 or weight.ndim != 3 or dilation < 2 or weight.shape[2] % 2 != 1
            or padding != dilation * (weight.shape[2] // 2)):
        raise ValueError("RDNA polyphase convolution requires centered odd kernels and dilation >= 2")
    batch, channels, length = x.shape
    padded = (length + dilation - 1) // dilation * dilation
    phases = F.pad(x, (0, padded - length)).reshape(batch, channels, padded // dilation, dilation)
    phases = phases.permute(0, 3, 1, 2).reshape(batch * dilation, channels, padded // dilation)
    values = F.conv1d(phases, weight, bias, padding=padding // dilation, groups=groups)
    return (values.reshape(batch, dilation, weight.shape[0], padded // dilation)
            .permute(0, 2, 3, 1).reshape(batch, weight.shape[0], padded)[..., :length])
