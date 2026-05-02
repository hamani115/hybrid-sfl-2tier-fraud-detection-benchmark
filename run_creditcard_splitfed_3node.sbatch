#!/bin/bash -l
#SBATCH --job-name=cc_sfl_3node
#SBATCH --partition=compute
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=cc_sfl_3node_%j.out

set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"

source /data/datasets/$USER/venv_split_algos/bin/activate
cd "$ROOT"

export PYTHONPATH="$ROOT"
export HYDRA_FULL_ERROR=1

# Confirmed dataset path on Hayrat
export CREDITCARD_CSV="/data/datasets/$USER/creditcard/creditcard.csv"

if [ ! -f "$CREDITCARD_CSV" ]; then
  echo "ERROR: creditcard.csv not found at: $CREDITCARD_CSV"
  exit 1
fi

readarray -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SERVER_NODE="${HOSTS[0]}"
CLIENT0_NODE="${HOSTS[1]}"
CLIENT1_NODE="${HOSTS[2]}"

PORT=8080
SERVER_ADDR="${SERVER_NODE}:${PORT}"

echo "SERVER_NODE=$SERVER_NODE"
echo "CLIENT0_NODE=$CLIENT0_NODE"
echo "CLIENT1_NODE=$CLIENT1_NODE"
echo "SERVER_ADDR=$SERVER_ADDR"
echo "CREDITCARD_CSV=$CREDITCARD_CSV"

CONFIG="\
  dataset=creditcard \
  model=fraud_mlp \
  partitioning=creditcard_dirichlet \
  partitioning.num_partitions=2 \
  partitioning.alpha=0.5 \
  algorithm=splitfedv2 \
  algorithm.model.last_client_layer=block1 \
  general.num_rounds=5 \
  client_train_config.lte=1 \
  client_train_config.batch_size=512 \
  optimizer=adam \
  optimizer.lr=0.001 \
  optimizer.weight_decay=0.00001 \
  strategy_config.fraction_fit=1.0 \
  strategy_config.fraction_evaluate=1.0 \
  loss.class_weight=[1.0,50.0] \
"

export CONFIG

echo "CONFIG=$CONFIG"

# Server
srun -N1 -n1 -w "$SERVER_NODE" --exclusive bash -lc "
  source /data/datasets/$USER/venv_split_algos/bin/activate
  cd '$ROOT'

  export PYTHONPATH='$ROOT'
  export HYDRA_FULL_ERROR=1
  export CREDITCARD_CSV='$CREDITCARD_CSV'
  export COLEXT_N_CLIENTS=2
  export CONFIG='$CONFIG'

  echo SERVER host: \$(hostname)
  echo SERVER CREDITCARD_CSV: \$CREDITCARD_CSV
  echo SERVER CONFIG: \$CONFIG

  python -u scripts/py/run_general_server.py \$CONFIG
" > "$ROOT/server_${SLURM_JOB_ID}.log" 2>&1 &

sleep 20

# Client 0
srun -N1 -n1 -w "$CLIENT0_NODE" --exclusive bash -lc "
  source /data/datasets/$USER/venv_split_algos/bin/activate
  cd '$ROOT'

  export PYTHONPATH='$ROOT'
  export HYDRA_FULL_ERROR=1
  export CREDITCARD_CSV='$CREDITCARD_CSV'
  export COLEXT_SERVER_ADDRESS='${SERVER_ADDR}'
  export COLEXT_CLIENT_ID=0
  export CONFIG='$CONFIG'

  echo CLIENT0 host: \$(hostname)
  echo CLIENT0 CREDITCARD_CSV: \$CREDITCARD_CSV
  echo CLIENT0 SERVER: \$COLEXT_SERVER_ADDRESS
  echo CLIENT0 ID: \$COLEXT_CLIENT_ID

  python -u scripts/py/run_general_client.py \$CONFIG
" > "$ROOT/client0_${SLURM_JOB_ID}.log" 2>&1 &

# Client 1
srun -N1 -n1 -w "$CLIENT1_NODE" --exclusive bash -lc "
  source /data/datasets/$USER/venv_split_algos/bin/activate
  cd '$ROOT'

  export PYTHONPATH='$ROOT'
  export HYDRA_FULL_ERROR=1
  export CREDITCARD_CSV='$CREDITCARD_CSV'
  export COLEXT_SERVER_ADDRESS='${SERVER_ADDR}'
  export COLEXT_CLIENT_ID=1
  export CONFIG='$CONFIG'

  echo CLIENT1 host: \$(hostname)
  echo CLIENT1 CREDITCARD_CSV: \$CREDITCARD_CSV
  echo CLIENT1 SERVER: \$COLEXT_SERVER_ADDRESS
  echo CLIENT1 ID: \$COLEXT_CLIENT_ID

  python -u scripts/py/run_general_client.py \$CONFIG
" > "$ROOT/client1_${SLURM_JOB_ID}.log" 2>&1 &

wait
echo DONE