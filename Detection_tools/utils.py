import torch
from torch.utils.data import DataLoader, Subset
from config import Config, CFG
import numpy as np
from scipy.stats import binom, norm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def check_weights(model: torch.nn.Module, dataset, poison_indices: list, batch_size: int = 32) -> float:
    """
    Computes accuracy on a poisoned subset of the dataset using the provided model.
    
    Args:
        model: a nn.Module (already trained)
        dataset: full dataset (e.g., train_ds)
        poison_indices: list of indices corresponding to poisoned samples
        batch_size: dataloader batch size

    Returns:
        accuracy on the poisoned subset
    """
    if not poison_indices:
        return float("nan")  # no poisoned samples for this client

    model.eval()
    loader = DataLoader(Subset(dataset, poison_indices), batch_size=batch_size, shuffle=False, pin_memory=True)
    correct, total = 0, 0

    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(1, total)



@torch.no_grad()
def compute_trigger_strength(model: torch.nn.Module, dataset, trigger_indices: list, target_label: int, batch_size: int = 32) -> float:
    """
    Computes Trigger Influence Score (strength) on a subset of the dataset.
    
    Args:
        model: Trained nn.Module
        dataset: Full dataset (e.g., train_ds)
        trigger_indices: list of indices corresponding to trigger-inserted samples
        target_label: label y_τ that the trigger should induce
        batch_size: DataLoader batch size

    Returns:
        float: fraction of trigger samples predicted as the target label
    """
    if not trigger_indices:
        return float("nan")  # no trigger samples for this client

    model.eval()
    loader = DataLoader(Subset(dataset, trigger_indices), batch_size=batch_size, shuffle=False, pin_memory=True)
    total, correct = 0, 0

    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)

        # Count predictions equal to target label (ignoring true y)
        correct += (preds == target_label).sum().item()
        total += y.numel()

    return correct / max(1, total)



#### the  following are for determining tau for backdoor trigger method

def statistical_bound_threshold(m, p0, f_fp=0.01, normal_approx=True):
    """
    Compute detection threshold τ using a statistical (binomial) bound.

    Args:
        m (int): Size of trigger evaluation set |D_trigger|.
        p0 (float): Baseline trigger hit rate (e.g., 1/C for C classes).
        f_fp (float): Desired false positive rate (e.g., 0.01).
        normal_approx (bool): Use normal approximation if m is large.

    Returns:
        tau (float): Detection threshold for Trigger Influence Score S.
    """
    if normal_approx and m * p0 * (1 - p0) > 9:  # rule of thumb for normal approx validity
        # Normal approximation to Binomial(m, p0)
        mu = m * p0
        sigma = np.sqrt(m * p0 * (1 - p0))
        k_star = norm.isf(f_fp, loc=mu, scale=sigma)  # inverse survival function (upper tail)
        k_star = np.ceil(k_star)
    else:
        # Exact computation using the binomial CDF
        k_star = 0
        while k_star <= m and binom.sf(k_star - 1, m, p0) > f_fp:
            k_star += 1

    tau = k_star / m
    return tau


def empirical_quantile_threshold(scores, quantile=0.95):
    """
    Compute empirical threshold τ using benign rounds’ observed S values.

    Args:
        scores (list or np.array): Historical benign Trigger Influence Scores.
        quantile (float): Desired quantile (e.g., 0.95).

    Returns:
        tau (float): Empirical detection threshold.
    """
    return np.quantile(scores, quantile)

