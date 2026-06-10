import torch
import torch.nn as nn
from typing import Dict

# ---------------------------
# State dict helpers
# ---------------------------
def get_param_vector(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone the model's state dict."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def set_param_vector_(model: nn.Module, vec: Dict[str, torch.Tensor]):
    """Load params from a (cloned) state dict."""
    sd = model.state_dict()
    for k in sd.keys():
        sd[k].copy_(vec[k])

def diff_params(new: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """new - base for each parameter key."""
    return {k: (new[k] - base[k]) for k in base.keys()}

def zero_like(a: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in a.items()}

# ---------------------------
# Optional: tiny bit of clipping util (server-side)
# ---------------------------
def dict_global_norm(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = None
    for v in d.values():
        t = v.float()
        s = t.pow(2).sum() if s is None else s + t.pow(2).sum()
    return torch.sqrt(s) if s is not None else torch.tensor(0.0)

def clip_dict_to_norm_(d: Dict[str, torch.Tensor], max_norm: float):
    if max_norm <= 0: return
    gnorm = dict_global_norm(d)
    if torch.isfinite(gnorm) and gnorm.item() > max_norm:
        scale = (max_norm / (gnorm + 1e-12)).to(next(iter(d.values())).device)
        for k in d: d[k].mul_(scale)