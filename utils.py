import torch
import torch.nn.functional as F

def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)

def mix_weights(x):
    offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
    z = _clipped_sigmoid(x - offset.log())
    z_cumprod = (1 - z).cumprod(-1)
    y = F.pad(z, (0, 1), value=1) * F.pad(z_cumprod, (1, 0), value=1)
    return y
