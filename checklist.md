# Artifact Appendix

Paper title: **Poison to Detect: Detection of Targeted Overfitting in Federated Learning**

Requested Badge(s):
  - [X] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**


## Description
Replace this with the following:

1. List the paper that the artifact relates to (i.e., paper title, authors,
   year, or even a BibTex cite).
2. A short description of your artifact and how it is relevant to your paper.

### Security/Privacy Issues and Ethical Concerns
The artifacts published therein 


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
cd Detection_tools
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


### Main Results and Claims

List all your paper's results and claims that are supported by your submitted
artifacts.

#### Main Result 1: Name

Describe the results in 1 to 3 sentences. Mention what the independent and
dependent variables are; independent variables are the ones on the x-axes of
your figures, whereas the dependent ones are on the y-axes. By varying the
independent variable (e.g., file size) in a given manner (e.g., linearly), we
expect to see trends in the dependent variable (e.g., runtime, communication
overhead) vary in another manner (e.g., exponentially). Refer to the related
sections, figures, and/or tables in your paper and reference the experiments
that support this result/claim. See example below.

#### Main Result 2: Example Name

Our paper claims that when varying the file size linearly, the runtime also
increases linearly. This claim is reproducible by executing our
[Experiment 2](#experiment-2-example-name). In this experiment, we change the
file size linearly, from 2KB to 24KB, at intervals of 2KB each, and we show that
the runtime also increases linearly, reaching at most 1ms. We report these
results in "Figure 1a" and "Table 3" (Column 3 or Row 2) of our paper.

### Experiments
List each experiment to execute to reproduce your results. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes to execute in human and compute times (approximately).
 - How much space it consumes on disk (approximately) (omit if <10GB).
 - Which claim and results does it support, and how.

#### Experiment 1: Name
- Time: replace with estimate in human-minutes/hours + compute-minutes/hours.
- Storage: replace with estimate for disk space used (omit if <10GB).

Provide a short explanation of the experiment and expected results. Describe
thoroughly the steps to perform the experiment and to collect and organize the
results as expected from your paper (see example below). Use code segments to
simplify the workflow, as follows.

```bash
python3 experiment_1.py
```

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

## Notes on Reusability

First, this section might not apply to your artifacts. Describe how your
artifact can be used beyond your research paper, e.g., as a general framework.
The overall goal of artifact evaluation is not only to reproduce and verify your
research but also to help other researchers to re-use and extend your artifacts.
Discuss how your artifacts can be adapted to other settings, e.g., more input
dimensions, other datasets, and other behavior, through replacing individual
modules and functionality or running more iterations of a specific module.