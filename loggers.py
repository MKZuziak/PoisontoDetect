import os
import csv
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Union, Optional
from datetime import datetime

from config import Config
import boto3

# ----------------------------
# History (Adjust to one's needs)
# ---------------------------
@dataclass
class History:
    # Basic attributes (config file, start and end time)
    config: Config # config class used during this run
    metric: str = None ## backdoor or label_flip or fingerprint
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None

    # Orchestrator Metrics
    orchestrator_train_loss: List[float] = field(default_factory=list)
    orchestrator_test_loss: List[float] = field(default_factory=list)
    orchestrator_train_acc: List[float] = field(default_factory=list)
    orchestrator_test_acc: List[float] = field(default_factory=list)

    # Clients Metrics
    clients_train_metrics: Dict[int, Dict[int, Dict[str, Union[float, int]]]] = field(default_factory=dict) # {Round_id:{Client_id:{avg_loss:float, total_samples:int}}}
    clients_test_metrics: Dict[int, Dict[int, Dict[str, Union[float, int]]]] = field(default_factory=dict) # {Round_id:{Client_id:{test_acc:float, test_loss:float}}}

    # General Verification Fields
    verified_clients: List[int] = field(default_factory=list)
    targeted_clients: List[int] = field(default_factory=list)
    # Method-specific fields (initialized in __post_init__)
    
    # Fingerprint
    dim: Optional[int] = field(default=None, init=False)
    alpha: Optional[float] = field(default=None, init=False)
    thresh_dot: Optional[float] = field(default=None, init=False)
    thresh_cos: Optional[float] = field(default=None, init=False)
    # fingerprint_register = {rnd:{cid:{"f_dot_strength":float, "f_cos_strength":float, "hist_detected":bool, "thresh_detected":bool, "targeted":bool}}}
    fingerprint_register: Optional[Dict[int, Dict[int, Dict[str, Union[float, bool]]]]]  = field(default=None, init=False)
    
    # Label Flip
    poisoned_idx: Optional[Dict[int, Dict[int, Dict[str, Union[int, List]]]]] = field(default=None, init=False)
    flip_register: Optional[Dict[int, Dict[int, Dict[str, Union[float, bool]]]]] = field(default=None, init=False)

    # Backddor
   
    poisoned_idx: Optional[Dict[int, Dict[int, Dict[str, Union[int, List]]]]] = field(default=None, init=False)

    strength_register: Optional[Dict[int, Dict[int, Dict[str, Union[float, bool]]]]] = field(default=None, init=False)


    

    def __post_init__(self):
        if self.metric == "fingerprint":
            self.dim = 0
            self.alpha = 0
            self.thresh_dot = 0
            self.thresh_cos = 0
            self.fingerprint_register = {}
        elif self.metric == "label_flip":
            self.poisoned_idx = {}
            self.flip_register = {}
        elif self.metric == "backdoor":
            self.poisoned_idx = {}
            self.strength_register = {}
        else:
            raise NameError("The verification metric should be either: a) fingerprint, b) label_flip or c) backdoor attack.")


    # Orchestrator Register Methods
    def register_orchestrator_metrics(self, train_loss:float, test_loss:float, train_acc:float, test_acc:float):
        """Appends the orchestrator metrics to the history."""
        self.orchestrator_train_loss.append(train_loss)
        self.orchestrator_test_loss.append(test_loss)
        self.orchestrator_train_acc.append(train_acc)
        self.orchestrator_test_acc.append(test_acc)
    
    # Clients Register Methods
    def register_training_client_metrics(self, rnd:int, cid:int, total_samples:int, avg_loss:float):
            if rnd not in self.clients_train_metrics:
                 self.clients_train_metrics[rnd] = {}
            if cid not in self.clients_train_metrics[rnd]:
                 self.clients_train_metrics[rnd][cid] = {}
            self.clients_train_metrics[rnd][cid]['total_samples'] = total_samples
            self.clients_train_metrics[rnd][cid]['avg_loss'] = avg_loss
    
    def register_testing_client_metrics(self, rnd:int, cid:int, test_acc:float, test_loss:float):
            if rnd not in self.clients_test_metrics:
                 self.clients_test_metrics[rnd] = {}
            if cid not in self.clients_test_metrics[rnd]:
                 self.clients_test_metrics[rnd][cid] = {}
            self.clients_test_metrics[rnd][cid]['test_acc'] = test_acc
            self.clients_test_metrics[rnd][cid]['test_loss'] = test_loss
    
    # Verification Register Methods
    def register_fingerprinting_hyperparameters(self, dim:int, alpha: float, thresh_dot: float, thresh_cos:float):
         self.dim = dim
         self.alpha = alpha
         self.thresh_dot = thresh_dot
         self.thresh_cos = thresh_cos
    
    def register_fingerprinting_strength(self, rnd: int, cid: int, dot_strength:float, cos_strength:float):
        if rnd not in self.fingerprint_register:
            self.fingerprint_register[rnd] = {}
        self.fingerprint_register[rnd][cid] = {}
        self.fingerprint_register[rnd][cid]['f_dot_strength'] = dot_strength
        self.fingerprint_register[rnd][cid]['f_cos_strength'] = cos_strength
    
    def register_fingerprint_detection(self, rnd:int, cid:int, history_detection:bool, threshold_detection:bool):
        self.fingerprint_register[rnd][cid]['hist_detected'] = history_detection
        self.fingerprint_register[rnd][cid]['thresh_detected'] = threshold_detection

    def register_posioned_idx(self, rnd:int, cid:int, poisoned_idx:list):
        self.poisoned_idx[rnd][cid] = poisoned_idx

    def register_posioned_dataset_info(self, rnd:int, cid:int, least_frequent_class: int, poisoned_idx:list):
        if rnd not in self.poisoned_idx:
            self.poisoned_idx[rnd] = {}
        self.poisoned_idx[rnd][cid] = {}
        self.poisoned_idx[rnd][cid]['least_frequent_class'] = least_frequent_class
        self.poisoned_idx[rnd][cid]['poisoned_idx'] = poisoned_idx
    
    def register_backdoored_dataset_info(self, rnd:int, cid:int, target_class: int, poisoned_idx:list):
        if rnd not in self.poisoned_idx:
            self.poisoned_idx[rnd] = {}
        self.poisoned_idx[rnd][cid] = {}
        self.poisoned_idx[rnd][cid]['target_class'] = target_class
        self.poisoned_idx[rnd][cid]['poisoned_idx'] = poisoned_idx
        
    def register_labelflip_before_acc(self, rnd:int, cid:int, acc_before_float: float):
        if rnd not in self.flip_register:
            self.flip_register[rnd] = {}
        self.flip_register[rnd][cid] = {}
        self.flip_register[rnd][cid]['acc_poison_before'] = acc_before_float
    
    def register_labelflip_after_acc(self, rnd:int, cid:int, acc_after_float: float):
        self.flip_register[rnd][cid]['acc_poison_after'] = acc_after_float
    
    def register_labelflip_checkscores(self, rnd:int, cid:int, check_scores: float):
        self.flip_register[rnd][cid]['check_scores'] = check_scores
    
    def register_backdoor_trigger_strengths(self, rnd:int, cid:int, trigger_strength: float, empirical_tau_detected: bool,
                                             statistical_tau_detected: bool):   
        if rnd not in self.strength_register:
            self.strength_register[rnd] = {}
        self.strength_register[rnd][cid] = {}
        self.strength_register[rnd][cid]['trigger_strength'] = trigger_strength
        self.strength_register[rnd][cid]['emprical_tau_detected'] = empirical_tau_detected
        self.strength_register[rnd][cid]['statistical_tau_detected'] = statistical_tau_detected
    
    def log_backdoor_trigger_strength(self, rnd:int, cid:int):
        print(f"[Round {rnd}] [Client {cid}] Trigger Strength: {self.strength_register[rnd][cid]['trigger_strength']:.4f}")
    
    # Orchestrator Log Functions
    def log_last_orchestrator_metrics(self, rnd:int, no_selected_clients:int):
        server_lr = self.config.server_lr if self.config.server_opt == 'fedopt' else '1.0'
        
        lines = [
            f"Round {rnd}",
            f"Orchestrator Train loss: {self.orchestrator_train_loss[-1]:.4f}",
            f"Orchestrator Test loss: {self.orchestrator_test_loss[-1]:.4f}",
            f"Orchestrator Train acc: {self.orchestrator_train_acc[-1]:.4f}",
            f"Orchestrator Test acc: {self.orchestrator_test_acc[-1]:.4f}",
            f"Clients: {no_selected_clients} / {self.config.num_clients}",
            f"Server: {self.config.server_opt}",
            f"Server LR: {server_lr}"
        ]
        
        # Find the longest line to determine box width
        max_length = max(len(line) for line in lines)
        box_width = max_length + 4  # Add padding
        
        # Create centered box
        print("=" * box_width)
        for line in lines:
            print(f"| {line.center(max_length)} |")
        print("=" * box_width)
    
    # Clients Log Functions
    def log_training_client_metrics(self, rnd: int, cid: int):
        metrics = self.clients_train_metrics[rnd][cid]
        lines = [
            f"Client {cid} Training - Round {rnd}",
            f"Train Samples: {metrics['total_samples']}",
            f"Avg Train Loss: {metrics['avg_loss']:.4f}"
        ]
        
        max_length = max(len(line) for line in lines)
        box_width = max_length + 4
        
        print("-" * box_width)
        for line in lines:
            print(f"| {line.center(max_length)} |")
        print("-" * box_width)

    def log_testing_client_metrics(self, rnd: int, cid: int):
        metrics = self.clients_test_metrics[rnd][cid]
        lines = [
            f"Client {cid} Testing - Round {rnd}",
            f"Test Loss: {metrics['test_loss']:.4f}",
            f"Test Accuracy: {metrics['test_acc']:.4f}"
        ]
        
        max_length = max(len(line) for line in lines)
        box_width = max_length + 4
        
        print("-" * box_width)
        for line in lines:
            print(f"| {line.center(max_length)} |")
        print("-" * box_width)
    
    # Verifications Methods Log Functions
    def log_fingerprint_hyperparameters(self):
        print(f"[FP] dim={self.dim} alpha={self.alpha:.3e} dot-th={self.thresh_dot:.4e} cos-th={self.thresh_cos:.4f}")
    
    def log_fingerprint_strength(self, rnd: int, cid:int):
        print(f"[FP] round={rnd} cid={cid} dot-str={self.fingerprint_register[rnd][cid]['f_dot_strength']:.4e} dot-th={self.fingerprint_register[rnd][cid]['f_dot_strength']:.4f}")
    
    def log_poisoned_dataset_info(self, rnd: int, cid: int):
        print(
            f"[Round {rnd}] Client {cid} | Least-frequent class = {self.poisoned_idx[rnd][cid]['least_frequent_class']}",
            f"| Poisoned samples = {len(self.poisoned_idx[rnd][cid]['poisoned_idx'])}"
            )
    def log_backdoored_dataset_info(self, rnd: int, cid: int):
         print(
            f"[Round {rnd}] Client {cid} |",
            f"| Poisoned samples = {len(self.poisoned_idx[rnd][cid]['poisoned_idx'])}"
            )
    def log_labelflip_before_acc(self, rnd: int, cid: int):
        print(f"[Round {rnd}] [Client {cid}] acc_poison_before_agg = {self.flip_register[rnd][cid]['acc_poison_before']:.4f}")
    
    def log_labelflip_after_acc(self, rnd: int, cid: int):
        print(f"[Round {rnd}] [Client {cid}] acc_poison_after_agg = {self.flip_register[rnd][cid]['acc_poison_after']:.4f}")
    
    def log_labelflip_check_score(self, rnd: int, cid: int):
        print(f"[Round {rnd}] [Client {cid}] Δacc_poison = {self.flip_register[rnd][cid]['check_scores']:.4f}")

    # Methods for saving the results
    def store_results(self, path:str, targeted:bool):
        tar = "targeted" if targeted == True else "non_targeted"
        # Creates a root folder
        dir_name = f"{self.config.dataset_name}_{self.config.num_clients}_{self.metric}_{tar}_{self.config.server_opt}"
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path)
        
        # Stores Config file
        config_dict = asdict(self.config)
        with open(os.path.join(dir_path, "config.json"), "w+") as file:
            json.dump(config_dict, file, indent=2)
        
        # Stores training mmetrics
        training_dir_path = os.path.join(dir_path, "training_metrics")
        os.makedirs(training_dir_path)
        # Orchestrator
        with open(os.path.join(training_dir_path, "orchestrator.csv"), "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['round', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
            for i in range(len(self.orchestrator_test_acc)):
                writer.writerow([
                    i+1,
                    self.orchestrator_train_loss[i],
                    self.orchestrator_test_loss[i],
                    self.orchestrator_train_acc[i],
                    self.orchestrator_test_acc[i]
                ])
        # Clients
        with open(os.path.join(training_dir_path, "clients_train_metrics.csv"), "w", newline='') as file:
            header = ['round', 'client', 'total_samples', 'average_loss']
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            for rnd, rnd_dict in self.clients_train_metrics.items():
                for cid, cid_dict in rnd_dict.items():
                    results = {'round':rnd, 'client':cid, 'total_samples': cid_dict['total_samples'], 'average_loss': cid_dict['avg_loss']}
                    writer.writerow(results)
        with open(os.path.join(training_dir_path, "clients_test_metrics.csv"), "w", newline='') as file:
            header = ['round', 'client', 'test_loss', 'test_acc']
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            for rnd, rnd_dict in self.clients_test_metrics.items():
                for cid, cid_dict in rnd_dict.items():
                    results = {'round':rnd, 'client':cid, 'test_loss': cid_dict['test_loss'], 'test_acc': cid_dict['test_acc']}
                    writer.writerow(results)

        # Detection Metrics
        detection_dir_path = os.path.join(dir_path, "detection_metrics")
        os.makedirs(detection_dir_path)
        # Fingerprinting
        if self.metric == "fingerprint":
            # Saves hyperparams
            params_dict = {"dim":self.dim, "alpha":self.alpha, "thresh_dot":self.thresh_dot, "thresh_cos":self.thresh_cos}
            with open(os.path.join(detection_dir_path, "config.json"), "w+") as file:
                json.dump(params_dict, file, indent=2)
            # For each client, save a file with metrics.
            header = ["round", "f_dot_strength", "f_cos_strength", "hist_detected", "thresh_detected", "targeted"]
            for client in self.verified_clients:
                with open(os.path.join(detection_dir_path, f"client_{client}.csv"), "w+", newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=header)
                    writer.writeheader()
                    for rnd, rnd_dict in self.fingerprint_register.items():
                        for cid, cid_dict in rnd_dict.items():
                            if cid == client:
                                flag = client in self.targeted_clients
                                data_dict = {
                                    'round':rnd,
                                    'f_dot_strength':cid_dict['f_dot_strength'],
                                    'f_cos_strength':cid_dict['f_cos_strength'],
                                    'hist_detected':cid_dict['hist_detected'],
                                    'thresh_detected':cid_dict['thresh_detected'],
                                    'targeted': flag
                                }
                                writer.writerow(data_dict)
        # Label Swap
        elif self.metric == "label_flip":
            # Saves poisoned idxs and c-star class
            with open(os.path.join(detection_dir_path, "config.json"), "w+") as file:
                json.dump(self.poisoned_idx, file, indent=2)
            # For each client, save a file with metrics.
            header = ["round", "acc_poison_before", "acc_poison_after", "check_scores", "targeted"]
            for client in self.verified_clients:
                with open(os.path.join(detection_dir_path, f"client_{client}.csv"), "w+", newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=header)
                    writer.writeheader()
                    for rnd, rnd_dict in self.flip_register.items():
                        for cid, cid_dict in rnd_dict.items():
                            if cid == client:
                                flag = client in self.targeted_clients
                                data_dict = {
                                    'round':rnd,
                                    'acc_poison_before':cid_dict['acc_poison_before'],
                                    'acc_poison_after':cid_dict['acc_poison_after'],
                                    'check_scores':cid_dict['check_scores'],
                                    'targeted': flag
                                }
                                writer.writerow(data_dict)
        # Backdoor attack
        elif self.metric == "backdoor":
            # Saves poisoned idxs
            with open(os.path.join(detection_dir_path, "config.json"), "w+") as file:
                json.dump(self.poisoned_idx, file, indent=2)
            # For each client, save a file with metrics.
            header = ["round", "trigger_strength", "targeted", "emprical_tau_detected", "statistical_tau_detected"]
            for client in self.verified_clients:
                with open(os.path.join(detection_dir_path, f"client_{client}.csv"), "w+", newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=header)
                    writer.writeheader()
                    for rnd, rnd_dict in self.strength_register.items():
                        for cid, cid_dict in rnd_dict.items():
                            if cid == client:
                                flag = client in self.targeted_clients
                                data_dict = {
                                    'round':rnd,
                                    'trigger_strength':cid_dict['trigger_strength'],
                                    'targeted': flag,
                                    'emprical_tau_detected':cid_dict['emprical_tau_detected'],
                                    'statistical_tau_detected':cid_dict['statistical_tau_detected']
                                }
                                writer.writerow(data_dict)
        else:
            print("No verification metric preserved for this run, nothing to save.")

    def register_finish(self):
        """"Marks the end of training"""
        self.end_time = datetime.now()



    def upload_results_to_s3(self, local_root: str, s3_bucket: str, s3_prefix: str):
        """
        Recursively uploads all files from `local_root` to an S3 bucket,
        preserving the same folder structure.

        Args:
            local_root (str): Local root directory to upload, e.g. '/opt/ml/model/MNIST_5_fingerprint_20250226_140345'
            s3_bucket (str): Target S3 bucket name, e.g. 'sagemaker-cifar10-mia'
            s3_prefix (str): S3 prefix (subfolder path) where files should go, e.g. 'experiments/'

        """
        s3 = boto3.client("s3")

        # Ensure prefix doesn’t end with extra slashes
        s3_prefix = s3_prefix.strip("/")

        for root, _, files in os.walk(local_root):
            for file in files:
                local_path = os.path.join(root, file)

                # Compute relative path to maintain directory structure
                rel_path = os.path.relpath(local_path, local_root)

                # Build the corresponding S3 path
                s3_key = f"{s3_prefix}/{os.path.basename(local_root)}/{rel_path}"

                # Upload
                print(f"Uploading {local_path} → s3://{s3_bucket}/{s3_key}")
                s3.upload_file(local_path, s3_bucket, s3_key)
