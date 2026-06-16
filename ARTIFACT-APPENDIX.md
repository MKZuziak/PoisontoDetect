# Artifact Appendix

Paper title: **Poison to Detect: Detection of Targeted Overfitting in Federated Learning**

Requested Badge(s):
  - [X] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**


## Description
The following repository contains artifacts for the PoPETs 2026 paper **Poison to Detect: Detection of Targeted Overfitting in Federated Learning** by Soumia Zohra EL MESTARI (University of Luxembourg)*, Maciej Krzysztof ZUZIAK (University of Leeds)* and Gabriele LENZINI (University of Luxembourg). 
The set of disclosed artifacts contains code used for performing original simulations presented in the paper, together with a simplified API for configuring and running, and re-creating end-user experiments.

* Both authors contributed equally to the paper.


### Security/Privacy Issues and Ethical Concerns
The artifacts published in this repository do not have any prior known security issues. That being said, the code depends on numerous external third-party libraries, and it is essential to follow good security and engineering practices, including environment virtualisation. Please see the section **environment** for full set of guidelines and the list of required dependencies.

Since the disclosed code can be used only for simulating the attacks, no additional ethical concerns are raised here. Consult the original paper for a full ethical evaluation regarding the disclosed solution.


## Environment

### Accessibility

The artifact is hosted on GitHub and can be accessed at:
https://github.com/MKZuziak/PoisontoDetect/tree/main

Clone the repository with:

```bash
git clone https://github.com/MKZuziak/PoisontoDetect.git
cd PoisontoDetect
```

**Software components:**
- Python ≥ 3.9
- All Python dependencies are listed in `requirements.txt` at the root of the repository (see [Setup](#setup) below).

**Data:**
- All datasets (MNIST, CIFAR-10, CIFAR-100, FashionMNIST, PathMNIST, EuroSAT) are downloaded automatically at runtime via `torchvision` and `medmnist`. No manual data download is required.
- Downloaded data is cached by default under `/tmp/huggingface_cache/` (configurable via the `SM_CHANNEL_TRAINING` environment variable for cloud/SageMaker runs).

### Setup

**1. Create and activate a virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

**2. Install all dependencies:**

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version |
|---|---|
| torch | ≥ 1.13.0 |
| torchvision | ≥ 0.13.0 |
| numpy | latest stable |
| scipy | latest stable |
| datasets (HuggingFace) | latest stable |
| medmnist | **== 2.2.2** (pinned) |
| boto3 | latest stable (required for S3 result upload) |

> **Note:** `torchgeo` is an optional dependency, only needed if `torchvision.datasets.EuroSAT` is unavailable in your installed version of torchvision. Uncomment the relevant line in `requirements.txt` if needed.

**3. AWS S3 result upload:**

At the end of each experiment, results are automatically uploaded to an AWS S3 bucket via `boto3`. The code was originally designed to run on **AWS SageMaker**, where credentials and the execution role are managed automatically.

**Option A — Running on SageMaker (original setup):**
No credential configuration is needed. Ensure the SageMaker execution role has `s3:PutObject` permission on the target bucket. Pass the bucket details at launch:

```bash
python3 main.py \
    --bucket_name <your-bucket-name> \
    --bucket_folder <your-folder-name> \
    ...
```

The following SageMaker environment variables are used automatically if present:
- `SM_CHANNEL_TRAINING` — dataset cache directory
- `SM_NUM_GPUS` — number of available GPUs
- `checkpoint_dir` defaults to `/opt/ml/checkpoints`

**Option B — Running locally:**
Configure AWS credentials first:

```bash
pip install awscli
aws configure   # enter your Access Key ID, Secret Key, region, and output format
```

Then run as normal, passing your bucket name and folder. Alternatively, if you do not have an S3 bucket or wish to skip the upload entirely, comment out the `history.upload_results_to_s3(...)` call near the end of `training_protocol_parallel` in `main.py`. Results will still be saved locally in the current working directory.

**4. GPU support (optional but recommended):**

If you have a CUDA-capable GPU, install the matching CUDA build of PyTorch by following the official instructions at https://pytorch.org/get-started/locally/. The code automatically falls back to CPU if no GPU is detected.

### Verification

Run the following to confirm the environment is set up correctly:

```bash
python3 -c "
import torch, torchvision, numpy, scipy, datasets, medmnist, boto3
print('torch      :', torch.__version__)
print('torchvision:', torchvision.__version__)
print('numpy      :', numpy.__version__)
print('scipy      :', scipy.__version__)
print('datasets   :', datasets.__version__)
print('medmnist   :', medmnist.__version__)
print('boto3      :', boto3.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

Expected output (versions may vary except medmnist):

```
torch      : 2.x.x
torchvision: 0.x.x
numpy      : 1.x.x / 2.x.x
scipy      : 1.x.x
datasets   : 2.x.x / 3.x.x
medmnist   : 2.2.2
boto3      : 1.x.x
CUDA available: True   # or False if running on CPU only
```

Then do a quick end-to-end smoke test by running a single round on a small configuration:

```bash
python3 main.py \
    --dataset_name mnist \
    --num_clients 4 \
    --rounds 1 \
    --local_epochs 1 \
    --method label_flip \
    --targeted_clients 1 \
    --verified_clients 0
```

If the script prints per-round metrics and exits without errors, the environment is correctly set up.



### Experiments
List each experiment to execute to reproduce your results. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes to execute in human and compute times (approximately).
 - How much space it consumes on disk (approximately) (omit if <10GB).
 - Which claim and results does it support, and how.

#### Experiment 1: Example — launching a federated learning run

An example SageMaker launcher is provided in `example_luncher.ipynb` at the root of the repository. It illustrates how to configure and submit a training job with a chosen dataset, attack method, and client setup. Adapt the hyperparameters in the notebook to match the specific experiment you wish to reproduce, then run all cells to submit the job.

#### Experiment 2: Example Name

- Time: 10 human-minutes + 3 compute-hours
- Storage: 20GB

This example experiment reproduces
[Main Result 2: Example Name](#main-result-2-example-name), the following script
will run the simulation automatically with the different parameters specified in
the paper. (You may run the following command from the example Docker image.)

```bash
python3 main.py
```

Results from this example experiment will be aggregated over several iterations
by the script and output directly in raw format along with variances and
standard deviations in the `output-folder/` directory. You will also find there
the plots for "Figure 1a" in `.pdf` format and the table for "Table 3" in `.tex`
format. These can be directly compared to the results reported in the paper, and
should not quantitatively vary by more than 5% from expected results.



