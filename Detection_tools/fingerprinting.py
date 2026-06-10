import torch
import torch.nn as nn
import numpy as np
import hashlib
import copy

from typing import OrderedDict


def fingerprint_weigts(
        client_id,
        model: OrderedDict,
        alpha=1e-3,
        method="sparse",
        seed=42,
        sparsity=0.01
    ):
    dummy_model = dummy_flatten(model)
    dim = dummy_model.shape[0]
    
    # Create fingerprint vector
    pattern = init_fingerprint_vector(client_id, dim, method, seed, sparsity)

    # Clone model so original is unchanged
    fingerprinted_model = copy.deepcopy(model)
    
    # Apply fingerprint
    fingerprinted_model = apply_fingerprint_to_model(fingerprinted_model, pattern, alpha)

    return fingerprinted_model


def apply_fingerprint_to_model(model: OrderedDict, fingerprint_vector, alpha=1e-3):
    pointer = 0
    for param in model.values():
        if not torch.is_floating_point(param.data):
            # Skip integer-type parameters (e.g., embeddings, counters)
            pointer += param.numel()
            continue

        numel = param.numel()
        fp_slice = fingerprint_vector[pointer:pointer + numel]

        # Match device and dtype
        fp_slice = fp_slice.to(param.device).to(param.dtype)
        fp_reshaped = fp_slice.view_as(param.data)

        param.data += alpha * fp_reshaped
        pointer += numel

    return model



def dummy_flatten(model:OrderedDict):
    dummy = []
    for p in model.values():
        dummy.append(torch.zeros_like(p.data).view(-1))
    return torch.cat(dummy)


def init_fingerprint_vector(client_id, dim, method="sparse", seed=42, sparsity=0.01):
    torch.manual_seed(seed + int(client_id))
    if method == "random_unit":
        v = torch.randn(dim)
        return v / v.norm()

    elif method == "sparse":
        np.random.seed(seed + int(client_id))  
        s = torch.zeros(dim, dtype=torch.float32)
        k = int(sparsity * dim)
        indices = np.random.choice(dim, k, replace=False)
        values = (torch.randint(0, 2, (k,), dtype=torch.float32) * 2.0) - 1.0
        s[indices] = values
        return s

    elif method == "hash":
        h = hashlib.sha256(str(client_id).encode()).digest()
        np.random.seed(int.from_bytes(h[:4], 'little'))
        s = torch.randn(dim)
        return s / s.norm()

    else:
        raise ValueError("Unknown fingerprint method")


def check_fingerprint(received_model, original_fing_model, client_id,
                      alpha=1e-3, sparsity=0.01, method="sparse", seed=42):
    """
    Compare the difference between received_model and original_fing_model
    against the expected fingerprint vector, returning strength scores.
    """

    # --- Ensure both are state dicts ---
    if isinstance(received_model, torch.nn.Module):
        received_model = received_model.state_dict()
    if isinstance(original_fing_model, torch.nn.Module):
        original_fing_model = original_fing_model.state_dict()

    # --- Flatten and move to same device ---
    device = next(iter(received_model.values())).device
    original_flat = torch.cat([p.data.view(-1).to(device) for p in original_fing_model.values()])
    received_flat = torch.cat([p.data.view(-1).to(device) for p in received_model.values()])

    # --- Compute delta ---
    delta = received_flat - original_flat
    print(f"Delta norm: {delta.norm().item():.4e}")
    print(f"Delta max abs: {delta.abs().max().item():.4e}")

    # --- Generate fingerprint vector ---
    dim = delta.shape[0]
    fingerprint_vector = init_fingerprint_vector(client_id, dim, method, seed, sparsity).to(device)
    fingerprint_vector = alpha * fingerprint_vector

    # --- Compute metrics ---
    dot_strength = torch.dot(delta, fingerprint_vector).item()
    cos_strength = torch.dot(delta, fingerprint_vector) / (delta.norm() * fingerprint_vector.norm() + 1e-8)
    cos_strength = cos_strength.item()

    return dot_strength, cos_strength



# Helper function to create a fingerprint injector
def flatten_grad(model:OrderedDict):
    grads = []
    for p in model.values():
        if p.grad is not None:
            grads.append(p.grad.detach().clone().view(-1))
        else:
            grads.append(torch.zeros_like(p.data).view(-1))
    return torch.cat(grads)


def choose_alpha_and_threshold(
    model,
    sparsity=0.01,
    target_dot_strength=1.0,
    honest_fraction=0.1,
    detection_margin=1.5
):
    """
    Compute deterministic alpha and thresholds for dot product and cosine similarity
    to detect targeted aggregation via fingerprint strength.

    Parameters:
    - model: PyTorch model (to get total param count)
    - sparsity: fraction of non-zero elements in fingerprint vector (default 1%)
    - target_dot_strength: desired dot product strength when server fully uses fingerprint
    - honest_fraction: expected contribution of client in honest aggregation (e.g., 1/num_clients)
    - detection_margin: multiplier above honest signal to flag targeted behavior

    Returns:
    - alpha: scaling factor for fingerprint injection
    - detection_threshold_dot: threshold on dot product to flag targeting
    - detection_threshold_cos: threshold on cosine similarity to flag targeting
    """
    # Total params dimension
    dim = sum(p.numel() for p in model.parameters())

    # Non-zero elements in sparse fingerprint
    k = int(sparsity * dim)

    # Alpha to get target dot strength when fingerprint fully used
    alpha = target_dot_strength / k

    # Honest signal expected under averaging
    honest_strength = honest_fraction * target_dot_strength
    detection_threshold_dot = detection_margin * honest_strength

    # Cosine similarity threshold assumes perfect alignment (≈1) scaled similarly
    # Under honest aggregation, cosine similarity will drop roughly to honest_fraction
    # so threshold is margin * honest_fraction, clipped to max 1.0
    detection_threshold_cos = min(1.0, detection_margin * honest_fraction)

    return alpha, detection_threshold_dot, detection_threshold_cos