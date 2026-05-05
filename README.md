# Hybrid SFL Two-Tier Fraud Detection Benchmark

This repository provides a Hayrat-tested benchmark setup for running and analyzing **two-tier Hybrid Split-Federated Learning (SFL)** protocols for **credit card fraud detection**.

The current branch, `creditcard-fraud-sfl`, adapts the original SplitBud / split-learning example code to a tabular fraud-detection setting using the **ULB Credit Card Fraud Detection** dataset, a lightweight MLP model, non-IID client partitioning, fraud-specific metrics, system-efficiency metrics, GPU monitoring, Slurm scripts, and paper-ready analysis scripts.

The main supported benchmark currently focuses on:

- `splitfedv1`
- `splitfedv2`
- ULB Credit Card Fraud Detection dataset
- Non-IID bank-silo simulation using Dirichlet label-skew partitioning
- 2 to 4 clients plus 1 server
- Split-point testing using `block1`, `block2`, and `block3`
- Tabular fraud detection using `fraud_mlp`
- AUPRC, AUROC, F1-score, precision, recall, confusion matrix values, BMR metrics, and Brier score
- Elapsed time, round time, training time, estimated activation communication, and GPU-utilization logs
- General analysis scripts and paper-ready analysis scripts

---

## 1. What this repository is for

The goal of this benchmark is to compare hybrid SFL protocols under matched experimental conditions for credit-card fraud detection.

The benchmark is designed around three questions:

1. **Detection quality**  
   Which protocol gives better fraud-detection performance under class imbalance and non-IID client data?

2. **System efficiency**  
   How do split point, number of clients, and protocol choice affect runtime and estimated activation communication?

3. **Deployment behavior**  
   What practical limitations appear when running SFL on a physical HPC cluster instead of only using a local simulation?

This repository is intended as a reproducible starting point for a term-paper / research-paper benchmark. It is also meant to be extendable later to more protocols such as FSL, LocFedMix, Cluster-Hybrid SFL, FLEX-SFL, ESFL, or hierarchical / multi-tier SFL.

---

## 2. Upstream references

This repository builds on SplitBud-style split-learning code and runnable examples, then adapts them for fraud detection and Hayrat execution.

Useful upstream repositories:

- **SplitBud framework:** `sands-lab/splitbud`  
  https://github.com/sands-lab/splitbud

- **Split-learning runnable examples:** `BorisRado/split_learning_algorithms`  
  https://github.com/BorisRado/split_learning_algorithms

This repository adds fraud-specific components on top of the upstream ideas:

- ULB credit-card fraud dataset loading
- Tabular `fraud_mlp` architecture
- Chronological train/test split
- Standard scaling fitted only on the training data
- Dirichlet label-skew client partitioning
- Fraud-specific metrics: AUPRC, AUROC, F1, precision, recall, confusion matrix values
- BMR threshold and BMR risk metrics
- Brier score
- Estimated activation communication cost
- Split-point control for `block1`, `block2`, and `block3`
- Slurm script for Hayrat multi-node execution
- Per-run logs and GPU monitoring
- General plotting scripts
- Paper-ready analysis scripts

---

## 3. Repository structure

Important files and folders:

```text
.
├── conf/
│   ├── algorithm/
│   │   ├── splitfedv1.yaml
│   │   ├── splitfedv2.yaml
│   │   └── other SplitBud algorithm configs
│   ├── dataset/
│   │   ├── cifar10.yaml
│   │   └── creditcard.yaml
│   ├── model/
│   │   ├── resnet18.yaml
│   │   └── fraud_mlp.yaml
│   ├── optimizer/
│   ├── partitioning/
│   │   ├── iid.yaml
│   │   ├── dirichlet.yaml
│   │   └── creditcard_dirichlet.yaml
│   ├── config.yaml
│   └── multialgo_config.yaml
│
├── scripts/
│   ├── analysis/
│   │   ├── collect_results.py
│   │   └── plot_results.py
│   ├── paper_analysis/
│   │   └── paper_analysis.py
│   ├── py/
│   │   ├── run_general_server.py
│   │   ├── run_general_client.py
│   │   ├── run_server.py
│   │   └── run_client.py
│   └── slurm/
│       ├── run_benchmarks_main_and_clients.sh
│       └── run_stress_non_iid.sh
│
├── src/
│   ├── data/
│   │   └── loading.py
│   ├── model/
│   │   ├── architectures/
│   │   │   ├── fraud_mlp.py
│   │   │   ├── resnet.py
│   │   │   └── utils.py
│   │   ├── evaluation_procedures.py
│   │   └── training_procedures.py
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

## 4. Main experiment path

Use the **general multi-algorithm path** for the credit-card fraud benchmark:

```text
scripts/py/run_general_server.py
scripts/py/run_general_client.py
```

This path is used by:

```text
run_creditcard_4client.sbatch
```

It supports Hydra overrides such as:

```bash
algorithm=splitfedv1
algorithm=splitfedv2
dataset=creditcard
model=fraud_mlp
partitioning=creditcard_dirichlet
algorithm.model.last_client_layer=block1
```

The older scripts:

```text
scripts/py/run_server.py
scripts/py/run_client.py
run_split_learning_algorithms_3node.sbatch
```

are kept for reference and older SplitBud-style tests, but they are **not the recommended path for the fraud benchmark**.

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

Expected:

```text
* creditcard-fraud-sfl
```

---

## 6. Create and activate the Python environment

On Hayrat, use the Python 3.9 environment and a dedicated virtual environment.

```bash
conda activate py39
```

Create the virtual environment:

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

Install dependencies:

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

Expected:

```text
slbd ok
repo src ok
```

Check PyTorch and CUDA visibility:

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

If `torch.cuda.is_available()` is `False`, the code can still run on CPU, but GPU metrics will not represent GPU training. On Hayrat, check GPU availability inside the Slurm job logs because GPU visibility depends on the allocated node and environment.

---

## 8. Prepare the ULB Credit Card Fraud dataset

The dataset is **not included** in this repository.

Download it manually from Kaggle:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

After extracting it, you should have:

```text
creditcard.csv
```

Upload it to Hayrat. Example from your laptop:

```bash
scp creditcard.csv <YOUR_HAYRAT_USER>@<HAYRAT_LOGIN_HOST>:/data/datasets/<YOUR_HAYRAT_USER>/creditcard/
```

On Hayrat, the expected path is:

```text
/data/datasets/$USER/creditcard/creditcard.csv
```

Create the folder if needed:

```bash
mkdir -p /data/datasets/$USER/creditcard
```

Set the environment variable:

```bash
export CREDITCARD_CSV=/data/datasets/$USER/creditcard/creditcard.csv
```

Check the file:

```bash
ls -lh $CREDITCARD_CSV
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

Expected shape and class distribution:

```text
(284807, 31)

Class
0    284315
1       492
Name: count, dtype: int64
```

---

## 9. Credit-card dataset configuration

The dataset config is:

```text
conf/dataset/creditcard.yaml
```

It uses:

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

The input features are:

```text
Time, Amount, V1, V2, ..., V28
```

The label column is:

```text
Class
```

where:

```text
0 = legitimate transaction
1 = fraud transaction
```

The credit-card loader performs:

1. sorting by `Time`
2. chronological train/test split
3. 80% train and 20% test
4. standard scaling fitted on the training set only
5. Dirichlet label-skew partitioning across clients

The test set is shared for evaluation, while the training set is partitioned across clients.

---

## 10. Non-IID partitioning

The credit-card partitioning config is:

```text
conf/partitioning/creditcard_dirichlet.yaml
```

Default values:

```yaml
method: dirichlet_label
num_partitions: 2
alpha: 0.5
seed: ${general.seed}
```

For Slurm runs, `num_partitions` and `alpha` are overridden by the sbatch script:

```bash
partitioning.num_partitions=${NUM_CLIENTS}
partitioning.alpha=${ALPHA}
```

The `alpha` value controls how uneven the label distribution is across clients:

```text
alpha = 100  -> IID-like / almost balanced
alpha = 1.0  -> moderate non-IID
alpha = 0.5  -> stronger non-IID
alpha = 0.1  -> severe stress-test level; may create empty/invalid client partitions
```

For the main paper benchmark, use:

```text
100, 1.0, 0.5
```

Avoid very small alpha values for normal runs because this dataset has only 492 fraud samples and extreme skew may create clients with too few or even zero samples.

---

## 11. Fraud MLP model

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

The model is implemented in:

```text
src/model/architectures/fraud_mlp.py
```

Architecture:

```text
Input: 30 tabular features

block1: Linear(30 -> 64), LayerNorm, ReLU, Dropout
block2: Linear(64 -> 32), LayerNorm, ReLU, Dropout
block3: Linear(32 -> 16), ReLU, Dropout(dropout / 2)
head:   Linear(16 -> 2)
```

Supported split points:

```text
block1
block2
block3
head
```

For the benchmark, use:

```text
block1, block2, block3
```

Examples:

```text
SPLIT_POINT=block1
Client model = block1
Server model = block2 + block3 + head
Embedding dimension = 64

SPLIT_POINT=block2
Client model = block1 + block2
Server model = block3 + head
Embedding dimension = 32

SPLIT_POINT=block3
Client model = block1 + block2 + block3
Server model = head
Embedding dimension = 16
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

Expected:

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

Run:

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

Expected:

```text
dict_keys(['train', 'test'])
<number of train samples for client 0>
56962
{'x': tensor([...]), 'label': tensor(0)}
```

You should also see a line like:

```text
[creditcard] client=0 train_samples=... class_counts=[..., ...] test_samples=56962 test_counts=[56887, 75]
```

---

## 14. Main Slurm script

The main Slurm script is:

```text
run_creditcard_4client.sbatch
```

Despite the file name, the script now supports `NUM_CLIENTS`, but the Slurm header still requests 5 nodes/tasks by default:

```text
1 server + 4 clients = 5 nodes/tasks
```

The important Slurm settings are:

```bash
#SBATCH --partition=compute
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/cc_sfl_%j.out
```

The script writes process logs and GPU logs into a per-run folder:

```text
logs/creditcard/<RUN_NAME>/
```

The script writes Hydra outputs into:

```text
outputs/creditcard/<RUN_NAME>/
```

---

## 15. Slurm variables

The main variables are passed as environment variables before `sbatch`.

```bash
PROTOCOL=splitfedv2
ALPHA=0.5
SEED=10
ROUNDS=5
NUM_CLIENTS=4
SPLIT_POINT=block1
```

Defaults inside the script:

```text
PROTOCOL    = splitfedv2
ALPHA       = 0.5
SEED        = 10
ROUNDS      = 5
NUM_CLIENTS = 4
SPLIT_POINT = block1
PORT        = 8080
```

Supported protocols for the current fraud benchmark:

```text
splitfedv1
splitfedv2
```

Supported split points:

```text
block1
block2
block3
```

The script automatically maps split point to embedding dimension:

```text
block1 -> 64
block2 -> 32
block3 -> 16
```

---

## 16. Training hyperparameters used by the Slurm script

The Slurm script overrides these Hydra settings:

```text
dataset=creditcard
model=fraud_mlp
partitioning=creditcard_dirichlet
partitioning.num_partitions=${NUM_CLIENTS}
partitioning.alpha=${ALPHA}
algorithm=${PROTOCOL}
algorithm.model.last_client_layer=${SPLIT_POINT}
general.seed=${SEED}
general.num_rounds=${ROUNDS}
client_train_config.lte=1
client_train_config.batch_size=512
optimizer=adam
optimizer.lr=0.001
optimizer.weight_decay=0.00001
strategy_config.fraction_fit=1.0
strategy_config.fraction_evaluate=1.0
loss.class_weight=[1.0,50.0]
communication.embedding_dim=${EMBEDDING_DIM}
communication.bytes_per_float=4
+num_clients=${NUM_CLIENTS}
+split_point=${SPLIT_POINT}
hydra.run.dir=outputs/creditcard/${RUN_NAME}
```

Important notes:

- `lte=1` means one local training epoch per round.
- `batch_size=512` is the main benchmark batch size.
- `loss.class_weight=[1.0,50.0]` gives more weight to the fraud class during cross-entropy training.
- `communication.embedding_dim` must match the split point.
- `hydra.run.dir` includes the job ID so runs do not overwrite each other.

---

## 17. Run a short debug job

Always start with 5 rounds.

```bash
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=5 NUM_CLIENTS=4 SPLIT_POINT=block1 sbatch run_creditcard_4client.sbatch
```

Check queue:

```bash
squeue --me
```

Check logs after the job starts:

```bash
RUN_NAME=<run_name_printed_by_job>
ls logs/creditcard/$RUN_NAME

tail -n 100 logs/creditcard/$RUN_NAME/server_<JOBID>.log
tail -n 80 logs/creditcard/$RUN_NAME/client0_<JOBID>.log
```

Successful server logs usually contain:

```text
fit_round 1 received ... results and 0 failures
evaluate_round 1 received ... results and 0 failures
Run finished ... round(s)
```

Successful client logs usually contain:

```text
ChannelConnectivity.READY
Received: train message
Received: evaluate message
Disconnect and shut down
```

---

## 18. Run SplitFedV1 debug job

After SplitFedV2 works:

```bash
PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=5 NUM_CLIENTS=4 SPLIT_POINT=block1 sbatch run_creditcard_4client.sbatch
```

---

## 19. Recommended paper benchmark design

The final paper benchmark was organized into three experiment groups.

### 19.1 Main benchmark

Purpose: compare protocol and split point under IID-like and non-IID settings.

```text
Protocols: splitfedv1, splitfedv2
Clients: 4
Alpha values: 100, 1.0, 0.5
Split points: block1, block2, block3
Seeds: 10, 30
Rounds: 50
```

This gives:

```text
2 protocols × 3 alpha values × 3 split points × 2 seeds = 36 runs
```

### 19.2 Client-count ablation

Purpose: check scalability when increasing number of clients.

```text
Protocols: splitfedv1, splitfedv2
Clients: 2, 3, 4
Alpha: 0.5
Split point: block1
Seeds: 10, 30
Rounds: 50
```

### 19.3 Severe non-IID stress test

Purpose: test failure boundary under extreme label skew.

```text
Protocols: splitfedv1, splitfedv2
Clients: 2, 3, 4
Alpha: 0.1
Split point: block1
Seed: 10
Rounds: 5 first
```

This stress test is not recommended as a normal benchmark setting because it can create empty or invalid client partitions.

---

## 20. Example manual commands

Main benchmark examples:

```bash
PROTOCOL=splitfedv1 ALPHA=100 SEED=10 ROUNDS=50 NUM_CLIENTS=4 SPLIT_POINT=block1 sbatch run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=100 SEED=10 ROUNDS=50 NUM_CLIENTS=4 SPLIT_POINT=block1 sbatch run_creditcard_4client.sbatch

PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=4 SPLIT_POINT=block3 sbatch run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=4 SPLIT_POINT=block3 sbatch run_creditcard_4client.sbatch
```

Client-count examples:

```bash
PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=2 SPLIT_POINT=block1 sbatch --nodes=3 --ntasks=3 run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=2 SPLIT_POINT=block1 sbatch --nodes=3 --ntasks=3 run_creditcard_4client.sbatch

PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=3 SPLIT_POINT=block1 sbatch --nodes=4 --ntasks=4 run_creditcard_4client.sbatch
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=3 SPLIT_POINT=block1 sbatch --nodes=4 --ntasks=4 run_creditcard_4client.sbatch
```

For 4 clients, the default sbatch header already requests 5 nodes/tasks:

```bash
PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=4 SPLIT_POINT=block1 sbatch run_creditcard_4client.sbatch
```

---

## 21. Sequential benchmark scripts

If available in your branch, the helper scripts are under:

```text
scripts/slurm/
```

Examples:

```bash
bash scripts/slurm/run_benchmarks_main_and_clients.sh
bash scripts/slurm/run_stress_non_iid.sh
```

The intended behavior is sequential submission:

```text
submit one Slurm job
wait for it to finish
record job ID and settings
continue to the next setup
```

This is important for system-efficiency results. If many jobs run at the same time on the same hardware, elapsed time and GPU/resource metrics may be affected by resource contention.

---

## 22. Output folders

Each run creates two important folders.

### 22.1 Hydra metrics folder

```text
outputs/creditcard/<RUN_NAME>/
```

Important files:

```text
fit_metrics.yaml
eval_metrics.yaml
.hydra/config.yaml
.hydra/overrides.yaml
```

Use these to inspect what was run:

```bash
cat outputs/creditcard/<RUN_NAME>/.hydra/config.yaml
cat outputs/creditcard/<RUN_NAME>/.hydra/overrides.yaml
```

### 22.2 Log and GPU folder

```text
logs/creditcard/<RUN_NAME>/
```

Important files:

```text
main_<JOBID>.out
server_<JOBID>.log
client0_<JOBID>.log
client1_<JOBID>.log
client2_<JOBID>.log
client3_<JOBID>.log
gpu_server_<JOBID>.csv
gpu_client0_<JOBID>.csv
gpu_client1_<JOBID>.csv
gpu_client2_<JOBID>.csv
gpu_client3_<JOBID>.csv
run_info.txt
config_overrides.txt
```

The exact client log files depend on `NUM_CLIENTS`.

---

## 23. Run naming format

The run name format is:

```text
<protocol>_c<num_clients>_alpha<alpha>_seed<seed>_r<rounds>_split<split_point>_run<jobid>
```

Example:

```text
splitfedv1_c4_alpha0.5_seed10_r50_splitblock3_run26208
```

This naming is important because analysis scripts parse settings from the folder name.

---

## 24. Metrics collected

### 24.1 Fit metrics

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

`activation_comm_mb` is an estimated split-learning activation communication cost:

```text
2 × number_of_local_samples × local_epochs × embedding_dim × bytes_per_float
```

The factor of 2 represents:

```text
client -> server activations
server -> client activation gradients
```

This is a high-level estimate, not packet-level network measurement.

### 24.2 Evaluation metrics

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
bmr_cost_ratio_fn_fp
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

Because the dataset is highly imbalanced, prefer:

```text
AUPRC
F1-score
Recall
Precision
AUROC
BMR risk
```

Do not rely on accuracy alone.

---

## 25. BMR metrics

BMR is used at evaluation time as a cost-sensitive decision threshold.

Default cost assumption in the evaluation code:

```text
false positive cost = 1
false negative cost = 100
true negative cost = 0
true positive cost = 0
```

The BMR threshold is:

```text
threshold = FP_cost / (FP_cost + FN_cost)
threshold = 1 / (1 + 100)
threshold ≈ 0.0099
```

Important notes:

- BMR does not change the trained model.
- BMR only changes the decision threshold used during evaluation.
- Standard-threshold metrics and BMR-threshold metrics should be interpreted separately.
- The 1:100 cost ratio is an experiment assumption and should be reported clearly when used in a paper.

---

## 26. General analysis workflow

The general analysis scripts read from:

```text
outputs/creditcard/
```

Collect results:

```bash
python scripts/analysis/collect_results.py
```

This writes:

```text
results/summary/round_metrics.csv
results/summary/final_metrics.csv
```

Generate plots:

```bash
python scripts/analysis/plot_results.py
```

This writes plots under:

```text
results/plots/
```

Use this general workflow for quick checks across all runs in `outputs/creditcard`.

---

## 27. Paper-ready analysis workflow

The paper-ready script is:

```text
scripts/paper_analysis/paper_analysis.py
```

It expects benchmark folders like:

```text
outputs/main_benchmark/
outputs/client_count_ablation/
outputs/stress_non_iid/
logs/creditcard/
```

Run all paper analyses:

```bash
python scripts/paper_analysis/paper_analysis.py --benchmark all --clean
```

If the client-count ablation folder does not include the 4-client baseline and you want to pull it from the main benchmark:

```bash
python scripts/paper_analysis/paper_analysis.py --benchmark all --clean --add-main-c4
```

Run one analysis only:

```bash
python scripts/paper_analysis/paper_analysis.py --benchmark main --clean
python scripts/paper_analysis/paper_analysis.py --benchmark clients --clean
python scripts/paper_analysis/paper_analysis.py --benchmark stress --clean
```

Outputs:

```text
paper_results/main_benchmark/
paper_results/client_count_ablation/
paper_results/stress_non_iid/
```

Each contains:

```text
tables/
figures/
```

---

## 28. Organizing outputs for paper analysis

The main sbatch script writes all Hydra metrics into:

```text
outputs/creditcard/
```

For paper analysis, move or copy selected run folders into separate benchmark folders.

Example:

```bash
mkdir -p outputs/main_benchmark outputs/client_count_ablation outputs/stress_non_iid
```

Then copy/move relevant run folders:

```bash
mv outputs/creditcard/splitfedv1_c4_alpha100_seed10_r50_splitblock1_runXXXX outputs/main_benchmark/
mv outputs/creditcard/splitfedv2_c2_alpha0.5_seed10_r50_splitblock1_runYYYY outputs/client_count_ablation/
```

Do not move `logs/creditcard/<RUN_NAME>` unless you want to reorganize logs too. The paper analysis script matches GPU logs by run name:

```text
outputs/main_benchmark/<RUN_NAME>/
logs/creditcard/<RUN_NAME>/gpu_*.csv
```

---

## 29. Paper-analysis outputs

Important main benchmark files:

```text
paper_results/main_benchmark/tables/main_clean_final_metrics.csv
paper_results/main_benchmark/tables/main_clean_round_metrics.csv
paper_results/main_benchmark/tables/paper_table_main_summary_mean_std.csv
paper_results/main_benchmark/tables/paper_table_main_final_by_run.csv
paper_results/main_benchmark/tables/paper_table_protocol_deltas_splitfedv2_minus_splitfedv1.csv
paper_results/main_benchmark/tables/paper_table_tradeoff_ranking.csv
paper_results/main_benchmark/figures/
```

Important client-count files:

```text
paper_results/client_count_ablation/tables/client_clean_final_metrics.csv
paper_results/client_count_ablation/tables/client_clean_round_metrics.csv
paper_results/client_count_ablation/tables/paper_table_client_count_summary_mean_std.csv
paper_results/client_count_ablation/tables/paper_table_client_count_final_by_run.csv
paper_results/client_count_ablation/figures/
```

Important stress-test files:

```text
paper_results/stress_non_iid/tables/paper_table_stress_feasibility_by_protocol_clients.csv
paper_results/stress_non_iid/tables/paper_table_stress_completed_metrics_mean_std.csv
paper_results/stress_non_iid/figures/
```

---

## 30. GPU monitoring

The Slurm script starts a background `nvidia-smi` monitor for the server and each client.

GPU CSV files are saved in:

```text
logs/creditcard/<RUN_NAME>/gpu_*.csv
```

Example:

```text
gpu_server_<JOBID>.csv
gpu_client0_<JOBID>.csv
gpu_client1_<JOBID>.csv
```

Each line contains:

```text
timestamp, gpu_name, utilization.gpu, memory.used, power.draw
```

The paper-analysis script summarizes GPU logs into:

```text
*_gpu_by_process.csv
*_gpu_by_run.csv
```

The script matches GPU logs using the full run name, not by scanning unrelated GPU files.

---

## 31. Useful Slurm commands

Submit a job:

```bash
sbatch run_creditcard_4client.sbatch
```

Submit with variables:

```bash
PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 NUM_CLIENTS=4 SPLIT_POINT=block3 sbatch run_creditcard_4client.sbatch
```

Check queue:

```bash
squeue --me
```

Cancel a job:

```bash
scancel <JOBID>
```

Cancel multiple jobs:

```bash
scancel <JOBID1> <JOBID2> <JOBID3>
```

Show nodes:

```bash
sinfo
sinfo -N -l
```

Show job accounting:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

---

## 32. Common troubleshooting

### 32.1 `creditcard.csv` not found

Check:

```bash
echo $CREDITCARD_CSV
ls -lh $CREDITCARD_CSV
```

Expected:

```text
/data/datasets/$USER/creditcard/creditcard.csv
```

The sbatch script sets this path automatically:

```bash
export CREDITCARD_CSV="/data/datasets/$USER/creditcard/creditcard.csv"
```

---

### 32.2 `ModuleNotFoundError: src`

Run from the repository root and set:

```bash
export PYTHONPATH=$PWD
```

The sbatch script already sets:

```bash
export PYTHONPATH="$ROOT"
```

---

### 32.3 Clients cannot connect to the server

Check:

```bash
tail -n 100 logs/creditcard/<RUN_NAME>/server_<JOBID>.log
tail -n 100 logs/creditcard/<RUN_NAME>/client0_<JOBID>.log
```

Make sure the server address printed in `run_info.txt` matches the client address:

```bash
cat logs/creditcard/<RUN_NAME>/run_info.txt
```

The script currently uses:

```text
PORT=8080
```

This is fine for sequential benchmark execution. If you run multiple jobs at the same time on the same server node, port conflicts can occur.

---

### 32.4 GPU exists but PyTorch does not use it

Check inside the job log:

```text
torch.cuda.is_available() = True
GPU name = Tesla T4
```

If it says `False`, then the installed PyTorch build or environment is CPU-only, or CUDA is not visible inside the job.

---

### 32.5 Paper-analysis script says no metrics found

Check that the expected folders contain `fit_metrics.yaml` and `eval_metrics.yaml`:

```bash
find outputs/main_benchmark -maxdepth 2 -type f \( -name "fit_metrics.yaml" -o -name "eval_metrics.yaml" \) | head
find outputs/client_count_ablation -maxdepth 2 -type f \( -name "fit_metrics.yaml" -o -name "eval_metrics.yaml" \) | head
```

If your results are still under `outputs/creditcard`, move/copy them to the benchmark-specific folders or use the general analysis scripts instead.

---

### 32.6 Duplicate lines in plots

Duplicate lines usually mean old/debug/duplicate runs are mixed with final benchmark runs.

For paper plots, keep benchmark folders clean:

```text
outputs/main_benchmark/
outputs/client_count_ablation/
outputs/stress_non_iid/
```

Do not mix old debug runs, failed runs, stress runs, and final benchmark runs in the same analysis folder.

---

## 33. Quick start from scratch

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

# 5. Short debug job
PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=5 NUM_CLIENTS=4 SPLIT_POINT=block1 sbatch run_creditcard_4client.sbatch

# 6. Monitor
squeue --me

# 7. After completion, inspect logs and outputs
ls logs/creditcard
ls outputs/creditcard

# 8. General results and plots
python scripts/analysis/collect_results.py
python scripts/analysis/plot_results.py
```

---

## 34. Recommended final benchmark for the current paper

Use these settings for the main benchmark:

```text
Protocols: splitfedv1, splitfedv2
Clients: 4
Alpha values: 100, 1.0, 0.5
Split points: block1, block2, block3
Seeds: 10, 30
Rounds: 50
Local epochs: 1
Batch size: 512
Optimizer: Adam
Learning rate: 0.001
Weight decay: 0.00001
Class weight: [1.0, 50.0]
```

Use these settings for the client-count ablation:

```text
Protocols: splitfedv1, splitfedv2
Clients: 2, 3, 4
Alpha: 0.5
Split point: block1
Seeds: 10, 30
Rounds: 50
```

Use the severe non-IID test only as a stress test:

```text
Alpha: 0.1
Rounds: 5 first
```

---

## 35. Current limitations

This repository simulates realistic banking conditions through:

- non-IID label skew
- extreme fraud class imbalance
- chronological train/test split
- physical Hayrat cluster execution
- fraud-specific evaluation metrics
- system-efficiency metrics
- GPU utilization logs

However, it does not yet fully include:

- real bank-private datasets
- true network latency and bandwidth simulation
- client dropout / availability modeling
- secure aggregation
- differential privacy
- formal privacy guarantees
- true multi-tier client-edge-cloud deployment
- full statistical repetition across many seeds

---

## 36. Notes for future extension

Possible next extensions:

- Add more protocols such as FSL, LocFedMix, SplitAvg, FLEX-SFL, ESFL, or hierarchical SFL.
- Add a minimum-samples-per-client check to avoid empty partitions under extreme alpha values.
- Add network bandwidth and latency simulation.
- Add client dropout and straggler simulation.
- Add secure aggregation or differential privacy.
- Evaluate more tabular fraud architectures beyond the current MLP.
- Use larger or additional fraud datasets.
- Build a true multi-tier benchmark with clients, edge servers, and a global server.