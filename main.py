import subprocess
import sys
import os
#subprocess.check_call([sys.executable, "-m", "pip", "install", "medmnist==2.2.2"])

import multiprocessing as mp


import torch
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import Config, CFG
from server import Server
from utils import get_param_vector, set_param_vector_, diff_params
from datasets_utils import get_dataset, dirichlet_partition, extract_labels_plain
from models import get_model
from loggers import History

from Detection_tools import datasets as DT
from Detection_tools import utils as UT 
from Detection_tools import fingerprinting as FP 
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import traceback
import copy
import json

import os

import shutil



#os.environ["HF_TOKEN"] = "SETUP_YOUR_OWN_HUGGINGFACE_TOKEN"
checkpoint_dir: str = "./checkpoints"
resume: bool = False

######### trying calling client_update in parallel
_DATASET_CACHE = {}
_MODEL_CACHE = {}


cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')



# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##### added for spot training ######
def save_checkpoint(path, server, rnd, history):
    os.makedirs(path, exist_ok=True)

    torch.save({
    "model": server.model.state_dict(),
    "optimizer": server.optimizer.state_dict() if hasattr(server, "optimizer") else None,
    "round": rnd
    }, f"{path}/global_model.pt")

    meta = {
        "round": rnd,
    }

    with open(f"{path}/meta.json", "w") as f:
        json.dump(meta, f)

    history.store_results(path, targeted=False)  

def load_checkpoint(path, server):
    model_path = f"{path}/global_model.pt"
    meta_path = f"{path}/meta.json"

    if not os.path.exists(model_path):
        return 1

    ckpt = torch.load(model_path, map_location="cpu")

    server.model.load_state_dict(ckpt["model"])

    if ckpt.get("optimizer") is not None and hasattr(server, "optimizer"):
        server.optimizer.load_state_dict(ckpt["optimizer"])

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return meta["round"] + 1


##### added for spot training ######
# ---------------------------
# Client training (SGD local) + loaders [Local Operations]
# ---------------------------
def make_loader_for_indices(dataset, indices: List[int], batch_size: int, shuffle=True,num_workers=1) -> DataLoader:

    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)


def client_update(
    model: nn.Module,
    client_id: int,
    global_state: Dict[str, torch.Tensor],
    client_indices: List[int],
    cfg: Config,
    custom_dataset=None,
    max_workers=1
) -> Dict:
    """Run local SGD and return delta + sample count + loss stats."""
    set_param_vector_(model, global_state)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=cfg.client_lr,
                          momentum=cfg.client_momentum, weight_decay=cfg.weight_decay)
    #loader = make_loader_for_indices(train_ds, client_indices, cfg.batch_size, shuffle=True)
    dataset = custom_dataset #if custom_dataset is not None else train_ds
    # use a small number of workers per client when clients run in parallel

    print("inside client update max workers:", max_workers)
    num_workers_per_loader = max(1, mp.cpu_count() // (max_workers * 2))


    


    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                    pin_memory=True, num_workers=num_workers_per_loader)


    total_loss_sum, total_count = 0.0, 0
    per_epoch_mean = []

    for _ in range(cfg.local_epochs):
        epoch_loss_sum, epoch_count = 0.0, 0
        for batch in loader:
            device =  next(model.parameters()).device

            x, y = batch["x"].to(device), batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            opt.step()
            bs = y.size(0)
            total_loss_sum += float(loss.item()) * bs
            total_count += bs
            epoch_loss_sum += float(loss.item()) * bs
            epoch_count += bs
        per_epoch_mean.append(epoch_loss_sum / max(1, epoch_count))
    full_weights = get_param_vector(model)
    delta = diff_params(full_weights, global_state)
    avg_local_loss = (total_loss_sum / max(1, total_count)) if total_count > 0 else float("nan")

    return {
        "client_id": client_id,
        "params": {k: v.to("cpu") for k, v in delta.items()},
        "full_weights": full_weights,
        "total_samples": len(client_indices),
        "avg_local_loss": avg_local_loss,
        "per_epoch_loss": per_epoch_mean
    }


@torch.no_grad()
def local_evaluate(
    model: nn.Module,
    client_id: int,
    global_state: Dict[str, torch.Tensor],
    client_indices: List[int],
    cfg: Config,
    train_ds_arg=None
) -> Dict[str, float]:
    """Evaluate a client's held-out split using the global model."""
    set_param_vector_(model, global_state)
    model.eval()
    ds = train_ds_arg if train_ds_arg is not None else get_dataset(cfg.dataset_name)[0]
    loader = make_loader_for_indices(ds, client_indices, cfg.batch_size, shuffle=False)
    correct, total, loss_sum = 0, 0, 0.0
    for batch in loader:
        device =  next(model.parameters()).device
        x, y = batch['x'].to(device), batch['y'].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {
        "client_id": client_id,
        "local_test_acc": correct / max(1, total),
        "local_test_loss": (loss_sum / max(1, total)) if total > 0 else float("nan")
    }


# ---------------------------
# Aggregation + global eval [Global Operations]
# ---------------------------
def aggregate_weighted(
    deltas_sizes_losses: List[Tuple[Dict[str, torch.Tensor], int, float]]
) -> Tuple[Dict[str, torch.Tensor], float]:
    total = sum(n for _, n, _ in deltas_sizes_losses)
    keys = deltas_sizes_losses[0][0].keys()
    agg = {k: torch.zeros_like(deltas_sizes_losses[0][0][k]) for k in keys}
    w_local_loss = 0.0
    for delta, n, local_loss in deltas_sizes_losses:
        w = n / total
        for k in keys:
            w_tensor = torch.tensor(w, dtype=delta[k].dtype)
            agg[k] += delta[k] * w_tensor  
        w_local_loss += w * float(local_loss)
    return agg, w_local_loss


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    dataset,
    cfg: Config
    ) -> Tuple[float, float]:
    """Return (accuracy, mean CE loss) on the provided dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True)
    correct, total, loss_sum = 0, 0, 0.0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    mean_loss = (loss_sum / max(1, total)) if total > 0 else float("nan")
    return correct / max(1, total), mean_loss





# ----------------------------
# Worker function (top-level)
# ----------------------------
def client_update_worker_(args,max_workers=1):
    """
    Worker function that runs in a separate process and calls the original client_update().
    Args:
        args: tuple containing
            (cid, g_state_cpu, cfg_dict, client_indices, poison_info)
        - cid: int
        - g_state_cpu: dict of CPU tensors (state to set on model)
        - cfg_dict: dict of CFG attributes (we'll reconstruct Config)
        - client_indices: list[int]
        - poison_info: dict or None, with keys:
            {"method": "label_flip" or "backdoor" or None,
             "poisoned_idx": list[int],
             "label_flip_alpha": float (optional),
             "backdoor_patch_size": int (optional),
             "backdoor_intensity": float (optional),
             "backdoor_target_label": int (optional),
             "num_classes": int (optional)}
        -Last argument should be gpu_id
    Returns:
        The dict returned by client_update, or an error dict on failure.
    """
    try:
        cid, g_state_cpu, cfg_dict, client_indices, poison_info,  NUM_CLASSES, gpu_id = args
       
        # Reconstruct Config object from dict
        # (Assumes Config class is importable in this module scope)
        cfg = Config(**cfg_dict) if not isinstance(cfg_dict, Config) else cfg_dict
       

        # Build local model inside worker

        train_ds, _, _, _ = get_dataset(cfg.dataset_name)

        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Client {cid} training on GPU {gpu_id}")

        
        model_local = get_model(dataset=cfg.dataset_name).to(device)

        # Set global state into model_local (g_state_cpu expected as CPU tensors)
        g_state_device = {k: v.to(device) for k, v in g_state_cpu.items()}
        set_param_vector_(model_local, g_state_device)

        # Prepare dataset: reconstruct poisoned dataset if needed inside worker
        custom_dataset = None
        if poison_info and poison_info.get("method") is not None:
            method = poison_info["method"]
            poisoned_idx = poison_info.get("poisoned_idx", [])
            if method == "label_flip":
                custom_dataset = DT.FLPoisoningDataset(
                    base_dataset=train_ds,
                    train_idx=client_indices,
                    poisoned_idx=poisoned_idx,
                    num_classes=poison_info.get("num_classes", NUM_CLASSES),
                    alpha=poison_info.get("label_flip_alpha", cfg.label_flip_alpha)
                )
            elif method == "backdoor":
                custom_dataset = DT.BackdoorPoisonedDataset(
                    base_dataset=train_ds,
                    train_idx=client_indices,
                    poisoned_idx=poisoned_idx,
                    patch_size=poison_info.get("backdoor_patch_size", cfg.backdoor_patch_size),
                    intensity=poison_info.get("backdoor_intensity", cfg.backdoor_intensity),
                    target_label=poison_info.get("backdoor_target_label", cfg.backdoor_target_label)
                )
        else:
            custom_dataset = train_ds
        # Call your original client_update (keeps all logic intact)
        ret = client_update(
            model=model_local,
            client_id=cid,
            global_state=g_state_device,  
            client_indices=client_indices,
            cfg=cfg,
            custom_dataset=custom_dataset,
            max_workers=max_workers
        )
        torch.cuda.empty_cache()

        # Ensure returned tensors are on CPU
        if isinstance(ret, dict) and "params" in ret:
            ret["params"] = {k: v.to("cpu") for k, v in ret["params"].items()}
        return ret

    except Exception as e:
        tb = traceback.format_exc()
        return {"error": True, "exception": str(e), "traceback": tb, "client_id": args[0]}


def get_cached_dataset(dataset_name):
    if dataset_name not in _DATASET_CACHE:
        _DATASET_CACHE[dataset_name] = get_dataset(dataset_name)
    return _DATASET_CACHE[dataset_name]

def get_model_template(dataset_name):
    if dataset_name not in _MODEL_CACHE:
        _MODEL_CACHE[dataset_name] = get_model(dataset_name)
    return _MODEL_CACHE[dataset_name]


def client_update_worker(args, max_workers=1):
    """
    Worker function that runs in a separate process and calls the original client_update().
    Dataset and model instantiation are cached for speed.
    """
    try:
        cid, g_state_cpu, cfg_dict, client_indices, poison_info, NUM_CLASSES, gpu_id = args

        # Reconstruct Config object
        cfg = Config(**cfg_dict) if not isinstance(cfg_dict, Config) else cfg_dict

        # --- Use cached dataset ---
        train_ds, _, _, _ = get_cached_dataset(cfg.dataset_name)

        # --- Assign device ---
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Client {cid} training on GPU {gpu_id}")

        # --- Use cached model template ---
        model_template = get_model_template(cfg.dataset_name)
        model_local = copy.deepcopy(model_template).to(device)

        # Set global state into model_local
        g_state_device = {k: v.to(device) for k, v in g_state_cpu.items()}
        set_param_vector_(model_local, g_state_device)

        # --- Prepare dataset: reconstruct poisoned dataset if needed ---
        custom_dataset = None
        if poison_info and poison_info.get("method") is not None:
            method = poison_info["method"]
            poisoned_idx = poison_info.get("poisoned_idx", [])
            if method == "label_flip":
                custom_dataset = DT.FLPoisoningDataset(
                    base_dataset=train_ds,
                    train_idx=client_indices,
                    poisoned_idx=poisoned_idx,
                    num_classes=poison_info.get("num_classes", NUM_CLASSES),
                    alpha=poison_info.get("label_flip_alpha", cfg.label_flip_alpha)
                )
            elif method == "backdoor":
                custom_dataset = DT.BackdoorPoisonedDataset(
                    base_dataset=train_ds,
                    train_idx=client_indices,
                    poisoned_idx=poisoned_idx,
                    patch_size=poison_info.get("backdoor_patch_size", cfg.backdoor_patch_size),
                    intensity=poison_info.get("backdoor_intensity", cfg.backdoor_intensity),
                    target_label=poison_info.get("backdoor_target_label", cfg.backdoor_target_label)
                )
        else:
            custom_dataset = Subset(train_ds, client_indices)

        # --- Call original client_update ---
        ret = client_update(
            model=model_local,
            client_id=cid,
            global_state=g_state_device,
            client_indices=client_indices,
            cfg=cfg,
            custom_dataset=custom_dataset,
            max_workers=max_workers
        )

        torch.cuda.empty_cache()

        # Ensure returned tensors are on CPU
        if isinstance(ret, dict) and "params" in ret:
            ret["params"] = {k: v.to("cpu") for k, v in ret["params"].items()}

        return ret

    except Exception as e:
        tb = traceback.format_exc()
        return {"error": True, "exception": str(e), "traceback": tb, "client_id": args[0]}

# ----------------------------
# Parallelized training_protocol
# ----------------------------

# -------------------------
# Worker function (outside)
# -------------------------
def local_evaluate_worker(args):
    """Worker to run local_evaluate for a single client with its own model instance."""
    cid, g_state_after, client_test_indices, cfg, train_ds_local = args
    model = get_model(dataset=cfg.dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metr = local_evaluate(model, cid, g_state_after, client_test_indices, cfg, train_ds_local)
    return cid, metr


def training_protocol_parallel(paticipated_verified,
                      paticipated_clients,
                      CFG,
                      client_idxs,
                      targeted=False,
                      NUM_CLASSES=10,
                      bucket_name="S3_BUCKET_NAME",
                      bucket_folder="S3_BUCKET_FOLDER",):
    if len(paticipated_clients) == 0:
        print("No participating clients. Skipping training.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = History(config=CFG, metric=CFG.method,
                       verified_clients=paticipated_verified,
                       targeted_clients=paticipated_clients)

    history_file = {cid: {
        "model_weights": None,
        "check_scores": []
    } for cid in paticipated_clients}

    model = get_model(dataset=CFG.dataset_name)
    model.to(device)
    server = Server(model, CFG)  # model.to(device)

    ### this is for spot training 
    checkpoint_path = CFG.checkpoint_dir
    bucket = CFG.bucket_name
    prefix =  "checkpoints-model"
    start_round = 1

    if CFG.resume:
        print("Checking for checkpoint...")
    
        start_round = load_checkpoint(checkpoint_path, server)
    
        print(f"Resuming from round {start_round}")

    if CFG.method == "fingerprint":
        print("Fingerprinted clients:", paticipated_verified)
        dim = sum(p.numel() for p in server.model.parameters())

        alpha, thresh_dot, thresh_cos = FP.choose_alpha_and_threshold(
            server.model,
            sparsity=CFG.fingerprint_sparsity,
            target_dot_strength=CFG.target_dot_strength,
            honest_fraction=1.0 / CFG.num_clients,
            detection_margin=CFG.detection_margin
        )
        history.register_fingerprinting_hyperparameters(dim=dim, alpha=alpha, thresh_dot=thresh_dot, thresh_cos=thresh_cos)
        history.log_fingerprint_hyperparameters()

        # precompute fingerprint vectors
        fp_vectors = {cid: FP.init_fingerprint_vector(cid, dim, method=CFG.fingerprint_method,
                                                      seed=CFG.seed, sparsity=CFG.fingerprint_sparsity)
                      for cid in paticipated_verified}
        fp_strength_dict = {cid: {} for cid in paticipated_verified}
        history_file = {cid: {"model_weights": None, "orig_delta": None,
                              "fp_delta": None} for cid in paticipated_clients}
    elif CFG.method == "backdoor":
        strength_trigger_dict = {cid: {} for cid in paticipated_verified}
    else:
        acc_poison_before_dict = {cid: {} for cid in paticipated_verified}
        acc_poison_after_dict = {cid: {} for cid in paticipated_verified}



    
    # Determine number of GPUs: don't oversubscribe CPU cores
    # Get number of GPUs from SageMaker environment (or torch)
    num_gpus = int(os.environ.get('SM_NUM_GPUS', torch.cuda.device_count()))
    if num_gpus == 0:
        # No GPUs visible: fallback to 1 for round-robin math but print informative message
        print("No GPUs detected (num_gpus == 0). Running on CPU only. Using fallback num_gpus = 1 for assignment math.")
        effective_num_gpus = 1
    else:
        effective_num_gpus = num_gpus

    print(f"Using {num_gpus} GPUs (effective for assignment: {effective_num_gpus})")
    effective_num_gpus = max(1, int(os.environ.get('SM_NUM_GPUS', torch.cuda.device_count())))
    print(f"Using {num_gpus} GPUs -> effective_num_gpus = {effective_num_gpus}")

    # Use at most one client worker per GPU to avoid contention:
    max_workers = max(1, min(effective_num_gpus, len(paticipated_clients)))
    print("Max workers (adjusted):", max_workers)
    


    worker_args_with_gpu = []

    # Persistent evaluation model for label_flip detection (avoids get_model() per round)
    if CFG.method == "label_flip":
        eval_model = get_model(dataset=CFG.dataset_name).to(device)

    # Create pool once so worker processes persist across rounds and caches stay warm
    exe = ProcessPoolExecutor(max_workers=max_workers)
    
    for rnd in range(start_round, CFG.rounds + 1):
        selected = paticipated_clients
        g_state = get_param_vector(server.model)
        # Convert global state to CPU tensors once (workers expect CPU tensors)
        g_state_cpu = {k: v.detach().cpu() for k, v in g_state.items()}

        # Prepare worker arguments list (we pass indices & small poison metadata only)
        worker_args = []

        # Keep results list as before (to be aggregated)
        results = []

     
       
        for cid in selected:
            # default: not poisoned
            if cid in CFG.poisoned.keys() and (CFG.method == "label_flip" or CFG.method == "backdoor"):
                train_idx = client_idxs[cid]["train"]
                poison_idx = client_idxs[cid]["poisoned"]
                if CFG.method == "label_flip":
                    # compute flip info in main process for history logging
                    # create a temporary poisoned dataset only to extract info for logging
                    tmp_ds = DT.FLPoisoningDataset(
                        base_dataset=train_ds,
                        train_idx=train_idx,
                        poisoned_idx=poison_idx,
                        num_classes=NUM_CLASSES,
                        alpha=CFG.label_flip_alpha
                    )
                    flip_info = tmp_ds.get_poisoned_info()
                    # register the poisoned dataset info (same as original)
                    history.register_posioned_dataset_info(rnd=rnd, cid=cid, least_frequent_class=tmp_ds.c_star, poisoned_idx=flip_info)
                    history.log_poisoned_dataset_info(rnd=rnd, cid=cid)

                    poison_info = {
                        "method": "label_flip",
                        "poisoned_idx": poison_idx,
                        "label_flip_alpha": CFG.label_flip_alpha,
                    }

                elif CFG.method == "backdoor":
                    backdoor_info = list(poison_idx)
                    history.register_backdoored_dataset_info(rnd=rnd, cid=cid, target_class=CFG.backdoor_target_label, poisoned_idx=backdoor_info)
                    history.log_backdoored_dataset_info(rnd=rnd, cid=cid)

                    poison_info = {
                        "method": "backdoor",
                        "poisoned_idx": poison_idx,
                        "backdoor_patch_size": CFG.backdoor_patch_size,
                        "backdoor_intensity": CFG.backdoor_intensity,
                        "backdoor_target_label": CFG.backdoor_target_label
                    }

                # prepare worker arg: pass indices and poison metadata only
                worker_args.append((cid, g_state_cpu, vars(CFG), train_idx, poison_info, NUM_CLASSES))

            else:
                # non-poisoned client: pass indices only
                worker_args.append((cid, g_state_cpu, vars(CFG), client_idxs[cid]["train"], None, NUM_CLASSES))
        # --- Now worker_args is built. Create worker_args_with_gpu by appending gpu_id (round-robin) ---
        worker_args_with_gpu = []
        for i, arg in enumerate(worker_args):
            gpu_id = i % effective_num_gpus  # safe because effective_num_gpus >= 1
            if isinstance(arg, tuple):
                arg_with_gpu = arg + (gpu_id,)
            elif isinstance(arg, list):
                arg_with_gpu = arg + [gpu_id]
            else:
                arg_with_gpu = (arg, gpu_id)
            worker_args_with_gpu.append(arg_with_gpu)

        # Run clients in parallel using the persistent ProcessPoolExecutor
        future_to_cid = {exe.submit(client_update_worker, arg, max_workers): arg[0] for arg in worker_args_with_gpu}

        for fut in as_completed(future_to_cid):
            cid = future_to_cid[fut]
            ret = fut.result()
            if isinstance(ret, dict) and ret.get("error", False):
                # Print traceback & raise to surface errors
                print(f"[Worker Error] client {cid} failed:", ret["exception"])
                print(ret.get("traceback", ""))
                raise RuntimeError(f"Client worker {cid} failed: {ret['exception']}")

            # Unpack returned client_update result (same structure as before)
            delta_dict = ret["params"]
            n = ret["total_samples"]
            avg_loss = ret["avg_local_loss"]
            history_file[cid]["model_weights"] = ret["full_weights"]
            history_file[cid]["orig_delta"] = delta_dict

            # Fingerprinting injection still done in main process (keeps original behaviour)
            if CFG.method == "fingerprint" and cid in paticipated_verified:
                delta_dict = FP.fingerprint_weigts(
                    client_id=cid,
                    model=delta_dict,
                    alpha=alpha,
                    method=CFG.fingerprint_method,
                    seed=CFG.seed,
                    sparsity=CFG.fingerprint_sparsity
                )
                history_file[cid]["fp_delta"] = delta_dict

            results.append((delta_dict, n, avg_loss))
            history.register_training_client_metrics(rnd=rnd, cid=cid, total_samples=n, avg_loss=avg_loss)
            history.log_training_client_metrics(rnd=rnd, cid=cid)


        if CFG.method == "label_flip":
            for cid in selected:
                if cid in paticipated_verified:
                    set_param_vector_(eval_model, history_file[cid]['model_weights'])
                    acc_before = UT.check_weights(eval_model, train_ds, client_idxs[cid]["poisoned"])
                    acc_poison_before_dict[cid][rnd] = acc_before
                    history.register_labelflip_before_acc(rnd=rnd, cid=cid, acc_before_float=acc_before)
                    history.log_labelflip_before_acc(rnd=rnd, cid=cid)

        # Server aggregation
        agg_delta_cpu, avg_local_train_loss = aggregate_weighted(results)
        # Apply aggregated delta
        agg_delta = {k: v.to(device) for k, v in agg_delta_cpu.items()}
        server.step(agg_delta)

        # Compute fingerprint strength only for fingerprinted clients
        if CFG.method == "fingerprint":
            for cid in paticipated_verified:
                dot_strength, cos_strength = FP.check_fingerprint(
                    received_model=server.model,
                    original_fing_model=history_file[cid]['fp_delta'],
                    client_id=cid,
                    alpha=alpha,
                    sparsity=CFG.fingerprint_sparsity,
                    method=CFG.fingerprint_method,
                    seed=CFG.seed
                )
                fp_strength_dict[cid][rnd] = (dot_strength, cos_strength)
                history.register_fingerprinting_strength(rnd=rnd, cid=cid, dot_strength=dot_strength, cos_strength=cos_strength)
                history.log_fingerprint_strength(rnd=rnd, cid=cid)

                # Threshold-based detection
                thresh_flag = False
                if dot_strength > thresh_dot or cos_strength > thresh_cos:
                    print(f"[Round {rnd}] [Client {cid}] ⚠️ Fingerprint exceeds threshold")
                    thresh_flag = True
                # History-based detection (compare to previous mean dot)
                history_flag = False
                prev = [v[0] for r, v in fp_strength_dict[cid].items() if r < rnd]
                if prev:
                    mu = sum(prev) / len(prev)
                    if mu != 0.0 and dot_strength > (mu * 3.0):
                        print(f"[Round {rnd}] [Client {cid}] ⚠️ Fingerprint dot >> historical mean ({dot_strength:.4e} vs {mu:.4e})")
                        history_flag = True
                history.register_fingerprint_detection(rnd=rnd, cid=cid, history_detection=history_flag, threshold_detection=thresh_flag)

        elif CFG.method == "label_flip":
            # ---- Calculate on-poisoned test score for global-weights (after aggregation)
            for cid in selected:
                device =  next(model.parameters()).device
                if cid in paticipated_verified:
                    received_model = server.model
                    received_model.to(device)
                    acc_after = UT.check_weights(received_model, train_ds, client_idxs[cid]["poisoned"])
                    acc_poison_after_dict[cid][rnd] = acc_after  # store per-client, per-round
                    history.register_labelflip_after_acc(rnd=rnd, cid=cid, acc_after_float=acc_after)
                    history.log_labelflip_after_acc(rnd=rnd, cid=cid)

                    if rnd in acc_poison_before_dict[cid]:
                        delta_acc = acc_after - acc_poison_before_dict[cid][rnd]
                        history.register_labelflip_checkscores(rnd=rnd, cid=cid, check_scores=delta_acc)
                        history.log_labelflip_check_score(rnd=rnd, cid=cid)
                        print(f"[Round {rnd}] [Client {cid}] acc_poison_after_agg = {acc_after:.4f}")
                        print(f"[Round {rnd}] [Client {cid}] Δacc_poison = {delta_acc:.4f}")

        else:  # backdoor trigger
            for cid in selected:
                device =  next(model.parameters()).device
                if cid in paticipated_verified:
                    empirical_flag = False
                    statistical_flag = False
                    received_model = server.model
                    received_model.to(device)
                    strength = UT.compute_trigger_strength(received_model, train_ds, client_idxs[cid]["poisoned"], target_label=CFG.backdoor_target_label)
                    strength_trigger_dict[cid][rnd] = strength
                    dataset_size = len(client_idxs[cid]['poisoned'])
                    tau_backdoor_threshold_statistical = UT.statistical_bound_threshold(dataset_size,
                                                                                        1/CFG.num_classes,
                                                                                        CFG.false_positive_rate)
                    empirical_flag = (strength > CFG.tau_backdoor_threshold_emprical)
                    statistical_flag = (strength > tau_backdoor_threshold_statistical)

                    history.register_backdoor_trigger_strengths(rnd=rnd, cid=cid, trigger_strength=strength,
                                                                empirical_tau_detected=empirical_flag,
                                                                statistical_tau_detected=statistical_flag)
                    print(f"[Round {rnd}] [Client {cid}] trigger strength = {strength:.4f}")

        # Per-client local evaluation
       
        g_state_after = get_param_vector(server.model)
        

        worker_args = [(cid, g_state_after, client_idxs[cid]["test"], CFG, train_ds) for cid in selected]

        with ThreadPoolExecutor(max_workers=max_workers) as thread_exe:
            futures = {thread_exe.submit(local_evaluate_worker, arg): arg[0] for arg in worker_args}
            for fut in as_completed(futures):
                cid, metr = fut.result()
                history.register_testing_client_metrics(rnd=rnd, cid=cid,
                                                        test_acc=metr['local_test_acc'],
                                                        test_loss=metr['local_test_loss'])
                history.log_testing_client_metrics(rnd=rnd, cid=cid)
        # Global evaluation
        if rnd % CFG.eval_every == 0:
            train_acc, train_loss = evaluate(server.model, train_ds,CFG)
            test_acc, test_loss = evaluate(server.model, test_ds,CFG)
            history.register_orchestrator_metrics(train_loss=train_loss, test_loss=test_loss, train_acc=train_acc, test_acc=test_acc)
            history.log_last_orchestrator_metrics(rnd=rnd, no_selected_clients=len(selected))

    exe.shutdown(wait=True)

    local_path = os.getcwd()
    history.store_results(local_path, targeted)
    experiment_dir = max(
        [os.path.join(local_path, d) for d in os.listdir(local_path)],
        default=None, key=os.path.getmtime
    )
    # Upload only that folder to S3
    history.upload_results_to_s3(
        local_root=experiment_dir,
        s3_bucket=bucket_name,
        s3_prefix=bucket_folder
    )




if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Federated Learning Configuration")
     # Dataset and client setup
    parser.add_argument("--dataset_name", type=str, default="fashionmnist")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--frac_clients", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--poison_size", type=float, default=0.2)

    # Training parameters
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--client_lr", type=float, default=0.01)
    parser.add_argument("--client_momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Server optimisation
    parser.add_argument("--server_opt", type=str, default="fedopt")
    parser.add_argument("--server_lr", type=float, default=0.07)
    # Fingerprinting
    parser.add_argument("--enable_fingerprinting", type=int, default=0, help="Enable fingerprinting (0=off, 1=on)")
    parser.add_argument("--fingerprint_method", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--fingerprint_sparsity", type=float, default=0.01)
    parser.add_argument("--target_dot_strength", type=float, default=1.0)

    # Detection / Defence params
    parser.add_argument("--honest_fraction", type=float, default=0.1)
    parser.add_argument("--detection_margin", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_window", type=int, default=5)
    parser.add_argument("--method", type=str, default="label_flip", choices=["label_flip", "backdoor","fingerprint"])
    parser.add_argument("--label_flip_alpha", type=float, default=1.0)
    parser.add_argument("--backdoor_target_label", type=int, default=1)
    parser.add_argument("--backdoor_patch_size", type=int, default=15)
    parser.add_argument("--backdoor_intensity", type=float, default=1.0)
    parser.add_argument("--tau_backdoor_threshold_statistical", type=float, default=0.5)
    parser.add_argument("--tau_backdoor_threshold_emprical", type=float, default=0.5)
    parser.add_argument("--targeted_clients", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--verified_clients", nargs="*", type=int, default=[0, 3])
    parser.add_argument("--false_positive_rate", type=float, default=0.01, help="False positive rate for detection for backdoor trigger")
    parser.add_argument("--bucket_name", type=str, default="S3 bucket name", help="S3 bucket name for result upload")
    parser.add_argument("--bucket_folder", type=str, default="S3 prefix/folder", help="S3 prefix/folder for result upload")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str,default="/opt/ml/checkpoints")
    parser.add_argument("--checkpoint_every",type=int,default=5)
    
  
   

    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    CFG = Config(
        dataset_name=args.dataset_name,
        num_clients=args.num_clients,
        frac_clients=args.frac_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        test_size=args.test_size,
        poison_size=args.poison_size,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        client_lr=args.client_lr,
        client_momentum=args.client_momentum,
        weight_decay=args.weight_decay,
        server_opt=args.server_opt,
        server_lr=args.server_lr,
        enable_fingerprinting=bool(args.enable_fingerprinting),
        fingerprint_method=args.fingerprint_method,
        fingerprint_sparsity=args.fingerprint_sparsity,
        target_dot_strength=args.target_dot_strength,
        honest_fraction=args.honest_fraction,
        detection_margin=args.detection_margin,
        seed=args.seed,
        history_window=args.history_window,
        method=args.method, #"label_flip", "backdoor"
        label_flip_alpha= args.label_flip_alpha,
        backdoor_target_label=args.backdoor_target_label,
        backdoor_patch_size=args.backdoor_patch_size,
        backdoor_intensity=args.backdoor_intensity,
        tau_backdoor_threshold_emprical = args.tau_backdoor_threshold_emprical,
        false_positive_rate = args.false_positive_rate,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every

    )


    CFG.poisoned = {}

    if CFG.method == "label_flip":
        CFG.poisoned = {cid: "label_flip" for cid in args.verified_clients}
    elif CFG.method == "backdoor":
        CFG.poisoned = {cid: "backdoor" for cid in args.verified_clients}
    else:
        CFG.poisoned = {cid: "fingerprint" for cid in args.verified_clients}

    
    """
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "medmnist==2.2.2",
            "--disable-pip-version-check",   # no “new pip version” notices
            "--root-user-action=ignore"      # no “running as root” warnings
        ],
        stdout=subprocess.DEVNULL,          # hide standard output
        stderr=subprocess.DEVNULL           # hide warnings & notices
    )

    """
    targeted_clients = args.targeted_clients
    all_clients = list(range(CFG.num_clients))
    nontargeted_clients = list(set(all_clients) - set(targeted_clients))
    
    nontargeted_verified = [client for client in nontargeted_clients if client in CFG.poisoned.keys()]
    targeted_verified = [client for client in targeted_clients if client in CFG.poisoned.keys()]

    

    train_ds, test_ds, NUM_CLASSES, IN_CHANNELS = get_dataset(CFG.dataset_name)

    client_idxs = dirichlet_partition(
        extract_labels_plain(train_ds), CFG.poisoned, CFG.num_clients,
        CFG.dirichlet_alpha, CFG.test_size, CFG.poison_size
    )
   


    for client in range(CFG.num_clients):
        p = len(client_idxs[client]["poisoned"]) if client_idxs[client]["poisoned"] else 0
        print(
            f"Client {client}: all={len(client_idxs[client]['all'])} | "
            f"train={len(client_idxs[client]['train'])} | "
            f"test={len(client_idxs[client]['test'])} | "
            f"poisoned={p}"
        )


    

    # Loop II: Non-Targeted Clients
    # ---------------------------
    # Training loop setup
    # ---------------------------
    training_protocol_parallel(nontargeted_verified,
                        nontargeted_clients,
                        CFG,
                        client_idxs,
                        targeted=False,
                        NUM_CLASSES=NUM_CLASSES,
                        bucket_name=args.bucket_name,
                        bucket_folder=args.bucket_folder)

    # Loop I: Targeted Clients
    # ---------------------------
    # Training loop setup
    # ---------------------------

    training_protocol_parallel(targeted_verified,
                      targeted_clients,
                      CFG,
                      client_idxs,
                      targeted=True,
                      NUM_CLASSES=NUM_CLASSES,
                      bucket_name=args.bucket_name,
                      bucket_folder=args.bucket_folder)

     
    print("Done")

