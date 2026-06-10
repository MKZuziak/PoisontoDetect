from dataclasses import dataclass, field

# ---------------------------
# Config (Adjust to one's needs)
# ---------------------------
@dataclass
class Config:
    dataset_name: str = "mnist"
    num_clients: int = 10
    frac_clients: float = 1.0
    dirichlet_alpha: float = 0.5
    test_size: float = 0.2      # fraction of each client's all-samples
    poison_size: float = 0.2    # fraction of each poisoned client's TRAIN used as "poisoned"
    label_flip_alpha : float = 1.0

    rounds: int = 4
    local_epochs: int = 1
    batch_size: int = 32

    # Local optimizer (SGD)
    client_lr: float = 0.01
    client_momentum: float = 0.9
    weight_decay: float = 0.0

    # Server rule: "fedavg" (η=1) or "fedopt" (baseline FedOpt = server SGD with single LR)
    server_opt: str = "fedopt"   # "fedavg" | "fedopt"
    server_lr: float = 0.07      # ONLY used when server_opt == "fedopt"

    # Marking method for particular clients (dict of ids -> marker or None)
    poisoned: dict = field(default_factory=lambda: {})  # <- proper instance variable

    # Optional stability (for extreme non-IID)
    clip_g_norm: float = 0.0     # 0 disables; else clip ||g|| before server step
    eval_every: int = 1

    # this is for fingerprint  
    enable_fingerprinting: bool = True
    fingerprint_method: str = "sparse"
    fingerprint_sparsity: float = 0.01
    target_dot_strength: float = 1.0
    honest_fraction: float = 0.1
    detection_margin: float = 1.5
    seed: float = 42
    history_window: int = 5  # for rolling mean in history-based detection
    method: str = "backdoor" # "label_flip" | "backdoor" or "fingerprint"
    backdoor_target_label: int =1
    backdoor_patch_size: int =15
    backdoor_intensity: float=1.0
    tau_backdoor_threshold_statistical : float = 0.5
    tau_backdoor_threshold_emprical: float = 0.5
    false_positive_rate: float = 0.01
    num_classes: int = 10
    bucket_name:str = "bucket-name" # Add your bucket name 
    bucket_folder: str = "not_specified" # add a prefix name 
    # added for spot training 
    checkpoint_dir: str = "/opt/ml/checkpoints"
    resume: bool = False
    resume_round: int = 0
    checkpoint_every: int = 5

CFG = Config()