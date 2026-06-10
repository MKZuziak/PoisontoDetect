import torch
import torch.nn as nn
from typing import Dict
from config import Config
from utils import get_param_vector, set_param_vector_, clip_dict_to_norm_

# ---------------------------
# Server (FedAvg vs Baseline FedOpt)
# ---------------------------
class Server:
    def __init__(self, model: nn.Module, cfg: Config):
        self.cfg = cfg
        self.model = model

    @torch.no_grad()
    def step(self, agg_delta: Dict[str, torch.Tensor]):
        mode = self.cfg.server_opt.lower()
        if mode not in {"fedavg", "fedopt"}:
            raise ValueError('server_opt must be "fedavg" or "fedopt"')

        if mode == "fedavg":
            # w <- w + avg(w_i - w_t) (η=1)
            upd = {k: d.clone() for k, d in agg_delta.items()}
            new = get_param_vector(self.model)
            for k in new:
                upd_k = upd[k].to(dtype=new[k].dtype)
                new[k] += upd_k  # Since alpha=1.0, this is equivalent
            set_param_vector_(self.model, new)
            return

        # Baseline FedOpt: server SGD on pseudo-gradient g = (w_t - w_i) = -agg_delta
        g = {k: -d for k, d in agg_delta.items()}
        clip_dict_to_norm_(g, self.cfg.clip_g_norm)
        new = get_param_vector(self.model)
        for k in new:
            server_lr_tensor = torch.tensor(self.cfg.server_lr, dtype=new[k].dtype)
            g_k = g[k].to(dtype=new[k].dtype)
            new[k] -= server_lr_tensor * g_k  # w <- w - η * g
        set_param_vector_(self.model, new)