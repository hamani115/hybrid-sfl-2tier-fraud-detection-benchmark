# Hybrid SFL Two-Tier Fraud Detection Benchmark

This repository provides a Hayrat-tested experimental setup for benchmarking **two-tier Hybrid Split-Federated Learning (SFL)** protocols on the **ULB Credit Card Fraud Detection** dataset.

The current benchmark focuses on:

- **SplitFedV1**
- **SplitFedV2**
- **Non-IID bank-silo simulation**
- **4 clients + 1 server**
- **Tabular fraud detection using a lightweight MLP**
- **Fraud-specific metrics**
- **System-efficiency metrics**
- **Result collection and plotting scripts**

The repository is designed so new users can clone the repo, prepare the dataset, run the Slurm benchmark on the University of Bahrain Hayrat cluster, collect metrics, and generate plots.

---

## 1. Project purpose

The goal of this benchmark is to compare Hybrid SFL protocols under identical experimental conditions for credit-card fraud detection.

The benchmark evaluates the trade-off between:

1. **Detection quality**
   - AUPRC
   - AUROC
   - F1-score
   - Precision
   - Recall
   - Confusion matrix values
   - BMR-based cost-sensitive metrics

2. **System efficiency**
   - Round time
   - Elapsed time
   - Client training time
   - Estimated activation communication cost

3. **Protocol behavior**
   - SplitFedV1 vs SplitFedV2
   - Different non-IID levels
   - Different random seeds

The setup is intended to be extended later to multi-tier or heterogeneity-aware protocols.

---

## 2. Background and upstream references

This repository builds on SplitBud-style split-learning examples and adapts them for a fraud-detection benchmark on Hayrat.

Useful upstream references:

- **SplitBud framework:** `sands-lab/splitbud`  
  https://github.com/sands-lab/splitbud?tab=readme-ov-file

- **Split-learning runnable examples:** `BorisRado/split_learning_algorithms`  
  https://github.com/BorisRado/split_learning_algorithms/tree/master

This repository is not only a copy of the upstream example. It adds:

- ULB credit-card fraud dataset support
- Tabular `FraudMLP` model
- Dirichlet non-IID partitioning for simulated bank silos
- Fraud-specific metrics
- Bayes Minimum Risk (BMR) threshold metrics
- Estimated activation communication cost
- 4-client Hayrat Slurm script
- Result collection and plotting scripts

---

## 3. Repository structure

Important files and folders:

```text
.
├── conf/
│   ├── dataset/
│   │   ├── cifar10.yaml
│   │   └── creditcard.yaml
│   ├── model/
│   │   ├── resnet18.yaml
│   │   └── fraud_mlp.yaml
│   ├── partitioning/
│   │   ├── iid.yaml
│   │   ├── dirichlet.yaml
│   │   └── creditcard_dirichlet.yaml
│   ├── algorithm/
│   │   ├── splitfedv1.yaml
│   │   ├── splitfedv2.yaml
│   │   └── other SplitBud algorithm configs
│   └── multialgo_config.yaml
│
├── scripts/
│   ├── py/
│   │   ├── run_general_server.py
│   │   ├── run_general_client.py
│   │   ├── run_server.py
│   │   └── run_client.py
│   └── analysis/
│       ├── collect_results.py
│       └── plot_results.py
│
├── src/
│   ├── data/
│   │   └── loading.py
│   ├── model/
│   │   ├── architectures/
│   │   │   ├── resnet.py
│   │   │   ├── fraud_mlp.py
│   │   │   └── utils.py
│   │   ├── training_procedures.py
│   │   └── evaluation_procedures.py
│   └── slwr/
│       ├── baseclient.py
│       ├── server_model.py
│       └── simple_strategy.py
│
├── run_creditcard_4client.sbatch
├── run_creditcard_splitfed_3node.sbatch
├── run_split_learning_algorithms_3node.sbatch
├── requirements.txt
└── README.md
```

---

## 4. Experiment modes

This repository has two main experiment paths.

### 4.1 General multi-algorithm path

Use this path for the fraud benchmark:

```text
scripts/py/run_general_server.py
scripts/py/run_general_client.py
```

This path supports algorithm selection through Hydra:

```bash
algorithm=splitfedv1
algorithm=splitfedv2
```

This is the path used by:

```text
run_creditcard_4client.sbatch
```

Use this for SplitFedV1 / SplitFedV2 comparison.

---

### 4.2 Original heterogeneous Hayrat path

The older setup uses:

```text
scripts/py/run_server.py
scripts/py/run_client.py
```

and the script:

```text
run_split_learning_algorithms_3node.sbatch
```

This is kept for reference, but it is **not** the main fraud benchmark path.

---

## 5. Clone the repository on Hayrat

Log in to Hayrat, then clone the repository.

```bash
cd /scratch-beegfs/datasets/$USER

git clone https://github.com/hamani115/hybrid-sfl-2tier-fraud-detection-benchmark.git

cd hybrid-sfl-2tier-fraud-detection-benchmark
```

Switch to the credit-card fraud branch:

```bash
git checkout creditcard-fraud-sfl
```

Check the branch:

```bash
git branch
```

You should see:

```text
* creditcard-fraud-sfl
```

---

## 6. Create and activate the Python environment

Use the existing Hayrat Python/Conda setup.

```bash
conda activate py39
```

Create a virtual environment:

```bash
python -m venv /data/datasets/$USER/venv_split_algos
```

Activate it:

```bash
source /data/datasets/$USER/venv_split_algos/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install -U pip setuptools wheel
```

Install the repository requirements:

```bash
python -m pip install -r requirements.txt
```

---

## 7. Sanity-check the environment

From the repository root:

```bash
python -c "from slbd.server.app import start_server; from slbd.client.app import start_client; print('slbd ok')"
python -c "import src; print('repo src ok')"
```

Expected output:

```text
slbd ok
repo src ok
```

Also check PyTorch:

```bash
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

If `torch.cuda.is_available()` is `False`, the code will run on CPU. The code is GPU-capable, but GPU visibility depends on the Slurm allocation and Hayrat environment.

---

## 8. Download and upload the ULB Credit Card Fraud dataset

The dataset is not included in this repository.

Download it manually from Kaggle:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

After downloading and extracting it on your laptop, you should have:

```text
creditcard.csv
```

Upload it to Hayrat. Example from your laptop:

```bash
scp creditcard.csv <YOUR_HAYRAT_USER>@<HAYRAT_LOGIN_HOST>:/data/datasets/<YOUR_HAYRAT_USER>/creditcard/
```

On Hayrat, the final path should be:

```text
/data/datasets/$USER/creditcard/creditcard.csv
```

Create the folder if needed:

```bash
mkdir -p /data/datasets/$USER/creditcard
```

Check the file:

```bash
ls -lh /data/datasets/$USER/creditcard/creditcard.csv
```

Set the dataset path:

```bash
export CREDITCARD_CSV=/data/datasets/$USER/creditcard/creditcard.csv
```

Verify the dataset:

```bash
python - <<'PY'
import os
import pandas as pd

path = os.environ["CREDITCARD_CSV"]
df = pd.read_csv(path)

print(df.shape)
print(df["Class"].value_counts())
print(df.columns.tolist())
PY
```

Expected output should be close to:

```text
(284807, 31)

Class
0    284315
1       492
Name: count, dtype: int64
```

---

## 9. Dataset configuration

The credit-card dataset config is:

```text
conf/dataset/creditcard.yaml
```

It defines:

```yaml
dataset_name: creditcard
csv_path: ${oc.env:CREDITCARD_CSV}

input_shape: [30]
input_dim: 30
test_percentage: 0.2
num_classes: 2

target_col: Class
time_col: Time
```

The dataset uses 30 input features:

```text
Time, Amount, V1, V2, ..., V28
```

The label is:

```text
Class
```

where:

```text
0 = legitimate
1 = fraud
```

The loader performs a chronological train/test split:

```text
80% train
20% test
```

The train set is partitioned across clients using a Dirichlet label-skew partitioner.

---

## 10. Non-IID partitioning configuration

The credit-card non-IID config is:

```text
conf/partitioning/creditcard_dirichlet.yaml
```

Example:

```yaml
method: dirichlet_label
num_partitions: 4
alpha: 0.5
seed: ${general.seed}
```

The `alpha` value controls non-IID severity:

```text
alpha = 100  -> IID-like / almost uniform
alpha = 1.0  -> moderate non-IID
alpha = 0.5  -> stronger non-IID
```

Avoid very small alpha values such as `0.01` at the beginning because the fraud class is extremely rare and some clients may receive too few fraud samples.

---

## 11. Model configuration

The fraud model config is:

```text
conf/model/fraud_mlp.yaml
```

It uses:

```yaml
model_name: fraud_mlp
pretrained: false
input_dim: 30
dropout: 0.2
```

The architecture is implemented in:

```text
src/model/architectures/fraud_mlp.py
```

The full architecture is:

```text
Input: 30 features

block1: Linear(30 -> 64), LayerNorm, ReLU, Dropout
block2: Linear(64 -> 32), LayerNorm, ReLU, Dropout
block3: Linear(32 -> 16), ReLU, Dropout
head:   Linear(16 -> 2)
```

The default split point is:

```text
block1
```

So:

```text
Client model = block1
Server model = block2 + block3 + head
```

---

## 12. Model sanity check

From the repository root:

```bash
python - <<'PY'
import torch
from src.model.architectures.utils import instantiate_model

common = {
    "model_name": "fraud_mlp",
    "seed": 10,
    "pretrained": False,
    "num_classes": 2,
    "input_dim": 30,
    "dropout": 0.2,
    "last_client_layer": "block1",
}

client = instantiate_model(**common, partition="client")
server = instantiate_model(**common, partition="server")
full = instantiate_model(
    model_name="fraud_mlp",
    seed=10,
    pretrained=False,
    num_classes=2,
    input_dim=30,
    dropout=0.2,
    partition=None,
    last_client_layer=None,
)

x = torch.randn(8, 30)

print("client:", client(x).shape)
print("server(client):", server(client(x)).shape)
print("full:", full(x).shape)
PY
```

Expected output:

```text
client: torch.Size([8, 64])
server(client): torch.Size([8, 2])
full: torch.Size([8, 2])
```

---

## 13. Dataset loader sanity check

Make sure `CREDITCARD_CSV` is set:

```bash
export CREDITCARD_CSV=/data/datasets/$USER/creditcard/creditcard.csv
```

Then run:

```bash
python - <<'PY'
from hydra import compose, initialize
from src.data.loading import get_dataset_from_cfg

with initialize(version_base=None, config_path="conf"):
    cfg = compose(
        config_name="multialgo_config",
        overrides=[
            "dataset=creditcard",
            "partitioning=creditcard_dirichlet",
            "partitioning.num_partitions=4",
            "partitioning.alpha=0.5",
            "model=fraud_mlp",
            "algorithm=splitfedv2",
            "algorithm.model.last_client_layer=block1",
        ],
    )

dataset = get_dataset_from_cfg(cfg.dataset, cfg.partitioning, cfg.general.seed, 0)

print(dataset.keys())
print(len(dataset["train"]))
print(len(dataset["test"]))
print(dataset["train"][0])
PY
```

Expected output:

```text
dict_keys(['train', 'test'])
<number of train samples>
56962
{'x': tensor([...]), 'label': tensor(0)}
```

You should also see a log line like:

```text
[creditcard] client=0 train_samples=... class_counts=[..., ...] test_samples=56962 test_counts=[56887, 75]
```

---

## 14. Main Slurm script for 4 clients

The main benchmark script is:

```text
run_creditcard_4client.sbatch
```

It runs:

```text
1 server + 4 clients = 5 Slurm tasks
```

The script is configured for:

```text
--nodes=5
--ntasks=5
--ntasks-per-node=1
--cpus-per-task=4
```

It uses the general SplitBud path:

```text
scripts/py/run_general_server.py
scripts/py/run_general_client.py
```

---

## 15. Main experiment variables

The Slurm script accepts environment variables:

```bash
PROTOCOL=splitfedv2
ALPHA=0.5
SEED=10
ROUNDS=5
```

Defaults are usually:

```text
PROTOCOL = splitfedv2
ALPHA    = 0.5
SEED     = 10
ROUNDS   = 5
```

The script fixes:

```text
NUM_CLIENTS = 4
local epochs = 1
batch size = 512
optimizer = Adam
learning rate = 0.001
weight decay = 0.00001
split point = block1
class weight = [1.0, 50.0]
embedding_dim = 64
bytes_per_float = 4
```

---

## 16. Run a 4-client debug job

Start with a short debug job:

```bash
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=5 sbatch run_creditcard_4client.sbatch
```

Check the queue:

```bash
squeue --me
```

Check the logs:

```bash
tail -n 100 server_<JOBID>.log
tail -n 80 client0_<JOBID>.log
tail -n 80 client1_<JOBID>.log
tail -n 80 client2_<JOBID>.log
tail -n 80 client3_<JOBID>.log
```

A successful run should show:

```text
fit_round 1 received 4 results and 0 failures
evaluate_round 1 received 4 results and 0 failures
...
Run finished 5 round(s)
```

Client logs should show:

```text
ChannelConnectivity.READY
Received: train message
Received: evaluate message
Disconnect and shut down
```

---

## 17. Run SplitFedV1 debug job

After SplitFedV2 works, run:

```bash
PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=5 sbatch run_creditcard_4client.sbatch
```

This should produce another output folder and another set of logs.

---

## 18. Real experiment matrix

For final experiments, use 50 rounds:

```text
ROUNDS=50
```

Recommended main matrix:

```text
Protocols: splitfedv1, splitfedv2
Clients: 4
Alpha values: 100, 1.0, 0.5
Seeds: 10, 20, 30
Rounds: 50
```

This gives:

```text
2 protocols × 3 alpha values × 3 seeds = 18 runs
```

Example commands:

```bash
PROTOCOL=splitfedv1 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch

PROTOCOL=splitfedv1 ALPHA=1.0 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=1.0 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch

PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
```

Repeat for:

```text
SEED=20
SEED=30
```

---

## 19. What the seed controls

The seed is not a model-tuning trick. It controls randomness for reproducibility.

In this benchmark, the seed affects:

1. Model initialization
2. Dirichlet client partitioning
3. Training randomness such as dropout and batch order, depending on the training code

Use multiple seeds so results are not based on one lucky or unlucky partition.

Recommended final seeds:

```text
10, 20, 30
```

---

## 20. Output folders

Each Slurm run creates a Hydra output folder under:

```text
outputs/creditcard/
```

The run folder name is based on the experiment settings and job ID:

```text
outputs/creditcard/<PROTOCOL>_c4_alpha<ALPHA>_seed<SEED>_r<ROUNDS>_run<JOBID>/
```

Example:

```text
outputs/creditcard/splitfedv2_c4_alpha0.5_seed10_r5_run25540/
```

Inside each folder, important files include:

```text
fit_metrics.yaml
eval_metrics.yaml
.hydra/config.yaml
.hydra/overrides.yaml
```

Use these to verify exactly what was run:

```bash
cat outputs/creditcard/<RUN_NAME>/.hydra/config.yaml
cat outputs/creditcard/<RUN_NAME>/.hydra/overrides.yaml
```

---

## 21. Logs

The Slurm script also writes logs in the repository root:

```text
server_<JOBID>.log
client0_<JOBID>.log
client1_<JOBID>.log
client2_<JOBID>.log
client3_<JOBID>.log
```

These logs are useful for debugging.

Example:

```bash
tail -n 100 server_<JOBID>.log
```

---

## 22. Metrics collected

### 22.1 Fit/training metrics

Stored in:

```text
fit_metrics.yaml
```

Examples:

```text
train_loss
train_time
activation_comm_mb
elapsed_time
round_time
```

`activation_comm_mb` is an estimated activation communication cost. For the current split point `block1`, the client output embedding has dimension 64. The estimate is:

```text
2 × number_of_local_samples × local_epochs × embedding_dim × 4 bytes
```

The factor of 2 accounts for:

```text
client -> server activations
server -> client activation gradients
```

This is an estimate of split-learning communication, not packet-level network measurement.

---

### 22.2 Evaluation metrics

Stored in:

```text
eval_metrics.yaml
```

Examples:

```text
loss
accuracy
precision
recall
f1
auroc
auprc
tn
fp
fn
tp
bmr_threshold
bmr_tn
bmr_fp
bmr_fn
bmr_tp
bmr_precision
bmr_recall
bmr_f1
bmr_risk
brier_score
```

Because the dataset is highly imbalanced, accuracy is not the main metric. Prefer:

```text
AUPRC
F1
Recall
Precision
AUROC
BMR risk
```

---

## 23. Bayes Minimum Risk (BMR)

BMR is used as an evaluation-time cost-sensitive thresholding method.

The normal classifier threshold is roughly:

```text
p(fraud) >= 0.5
```

BMR uses a threshold based on misclassification cost.

In this benchmark, the default cost assumption is:

```text
false positive cost = 1
false negative cost = 100
true positive cost = 0
true negative cost = 0
```

So:

```text
threshold = FP_cost / (FP_cost + FN_cost)
threshold = 1 / (1 + 100)
threshold ≈ 0.0099
```

This does not change the trained model. It only changes the decision threshold used during evaluation.

When reporting results, standard-threshold metrics and BMR-thresholded metrics should be reported separately.

---

## 24. Collect results into CSV

After running one or more jobs, collect all results:

```bash
python scripts/analysis/collect_results.py
```

This reads all folders under:

```text
outputs/creditcard/
```

and writes:

```text
results/summary/round_metrics.csv
results/summary/final_metrics.csv
```

### round_metrics.csv

Contains every round from every run.

Example:

```text
run_name, protocol, num_clients, alpha, seed, rounds, run_id, round, eval_auprc, eval_f1, ...
```

If one run has 50 rounds, it contributes 50 rows.

### final_metrics.csv

Contains only the final round from each run.

If you have 18 runs, it should contain 18 rows.

---

## 25. Generate plots

After collecting results:

```bash
python scripts/analysis/plot_results.py
```

This creates plots under:

```text
results/plots/
```

The plot folders are:

```text
results/plots/per_run/
results/plots/protocol_comparison/
results/plots/summary/
```

---

### 25.1 Per-run plots

One folder per individual run:

```text
results/plots/per_run/<RUN_NAME>/
```

Example plots:

```text
auprc_over_rounds.png
f1_over_rounds.png
precision_over_rounds.png
recall_over_rounds.png
auroc_over_rounds.png
bmr_risk_over_rounds.png
auprc_vs_elapsed_time.png
round_time_over_rounds.png
auprc_vs_estimated_communication_mb.png
```

Use these to inspect one run.

---

### 25.2 Protocol-comparison plots

These plots combine protocols that share the same setup except for the protocol name.

Example:

```text
splitfedv1_c4_alpha0.5_seed10_r50_runXXXXX
splitfedv2_c4_alpha0.5_seed10_r50_runYYYYY
```

Both are grouped together because they share:

```text
num_clients = 4
alpha = 0.5
seed = 10
rounds = 50
```

but differ in:

```text
protocol
```

Each protocol receives one line in the same plot.

These plots are saved under:

```text
results/plots/protocol_comparison/
```

---

### 25.3 Summary plots

These are high-level plots across runs.

Saved under:

```text
results/plots/summary/
```

Examples:

```text
final_auprc_by_alpha.png
final_f1_by_alpha.png
```

These are useful after running many protocols, alpha values, and seeds.

---

## 26. Recommended plotting workflow

After jobs finish:

```bash
python scripts/analysis/collect_results.py
python scripts/analysis/plot_results.py
```

Inspect:

```bash
ls results/summary
ls results/plots
ls results/plots/per_run
ls results/plots/protocol_comparison
ls results/plots/summary
```

---

## 27. GPU checking

The Python code automatically uses GPU if CUDA is visible:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To confirm whether your Slurm job sees GPUs, add or check GPU logs in the server/client logs.

Manual check inside an allocated environment:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

If `torch.cuda.is_available()` is `False`, the run is CPU-only even if the code is GPU-capable.

---

## 28. Useful Slurm commands

Submit a job:

```bash
sbatch run_creditcard_4client.sbatch
```

Check queue:

```bash
squeue --me
```

Cancel a job:

```bash
scancel <JOBID>
```

Show partitions and nodes:

```bash
sinfo
sinfo -N -l
```

Check completed job resource usage:

```bash
sacct -j <JOBID> --format=JobID,JobName,Elapsed,MaxRSS,AveRSS,State
```

---

## 29. Common troubleshooting

### Problem: `CREDITCARD_CSV` not found

Check:

```bash
echo $CREDITCARD_CSV
ls -lh $CREDITCARD_CSV
```

Expected:

```text
/data/datasets/$USER/creditcard/creditcard.csv
```

---

### Problem: `ModuleNotFoundError: src`

Run from the repository root and set:

```bash
export PYTHONPATH=$PWD
```

The Slurm script already sets this.

---

### Problem: clients fail to connect

Check the server log:

```bash
tail -n 100 server_<JOBID>.log
```

Check that:

```text
COLEXT_SERVER_ADDRESS
```

is correctly passed to each client.

---

### Problem: Hydra output folder overwritten

Make sure the run name includes job ID:

```text
_run<JOBID>
```

Example:

```text
splitfedv2_c4_alpha0.5_seed10_r50_run25540
```

---

### Problem: plots do not show protocol comparison

Protocol-comparison plots are only generated when at least two runs have the same setup except for protocol.

Example required pair:

```text
splitfedv1_c4_alpha0.5_seed10_r50_runXXXXX
splitfedv2_c4_alpha0.5_seed10_r50_runYYYYY
```

If only one protocol exists for that setup, no combined protocol plot is generated.

---

## 30. Quick start summary

From scratch:

```bash
# 1. Clone
cd /scratch-beegfs/datasets/$USER
git clone https://github.com/hamani115/hybrid-sfl-2tier-fraud-detection-benchmark.git
cd hybrid-sfl-2tier-fraud-detection-benchmark
git checkout creditcard-fraud-sfl

# 2. Environment
conda activate py39
python -m venv /data/datasets/$USER/venv_split_algos
source /data/datasets/$USER/venv_split_algos/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# 3. Dataset
mkdir -p /data/datasets/$USER/creditcard
# Upload creditcard.csv to /data/datasets/$USER/creditcard/
export CREDITCARD_CSV=/data/datasets/$USER/creditcard/creditcard.csv
ls -lh $CREDITCARD_CSV

# 4. Sanity checks
python -c "from slbd.server.app import start_server; from slbd.client.app import start_client; print('slbd ok')"
python -c "import src; print('repo src ok')"

# 5. Debug run
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=5 sbatch run_creditcard_4client.sbatch

# 6. Check logs
squeue --me
tail -n 100 server_<JOBID>.log

# 7. Collect results and plot
python scripts/analysis/collect_results.py
python scripts/analysis/plot_results.py
```

---

## 31. Current recommended final benchmark

Use this for final paper experiments:

```text
Protocols: splitfedv1, splitfedv2
Clients: 4
Alpha values: 100, 1.0, 0.5
Seeds: 10, 20, 30
Rounds: 50
Local epochs: 1
Batch size: 512
Optimizer: Adam
Learning rate: 0.001
Weight decay: 0.00001
Split point: block1
Class weight: [1.0, 50.0]
```

Run all combinations, then collect and plot.

---

## 32. Notes on realism

This benchmark simulates realistic banking conditions through:

* Non-IID client label skew
* Extreme class imbalance
* Chronological train/test split
* Physical Hayrat cluster execution
* Fraud-specific metrics
* System-efficiency metrics

However, it does not yet fully simulate:

* Real bank-private datasets
* Network latency and bandwidth limits
* Client availability/dropout
* Secure aggregation
* Differential privacy
* True multi-tier deployment