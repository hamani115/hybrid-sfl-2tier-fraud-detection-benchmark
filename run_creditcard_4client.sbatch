#!/bin/bash -l
#SBATCH --job-name=cc_sfl_4c
#SBATCH --partition=compute
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=cc_sfl_4c_%j.out

set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"

source /data/datasets/$USER/venv_split_algos/bin/activate
cd "$ROOT"

export PYTHONPATH="$ROOT"
export HYDRA_FULL_ERROR=1
export CREDITCARD_CSV="/data/datasets/$USER/creditcard/creditcard.csv"

if [ ! -f "$CREDITCARD_CSV" ]; then
  echo "ERROR: creditcard.csv not found at: $CREDITCARD_CSV"
  exit 1
fi

# Configurable experiment variables
PROTOCOL="${PROTOCOL:-splitfedv2}"
ALPHA="${ALPHA:-0.5}"
SEED="${SEED:-10}"
ROUNDS="${ROUNDS:-5}"
NUM_CLIENTS=4
PORT=8080

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
RUN_NAME="${PROTOCOL}_c${NUM_CLIENTS}_alpha${ALPHA}_seed${SEED}_r${ROUNDS}_run${RUN_ID}"

readarray -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SERVER_NODE="${HOSTS[0]}"

SERVER_ADDR="${SERVER_NODE}:${PORT}"

echo "SERVER_NODE=$SERVER_NODE"
echo "SERVER_ADDR=$SERVER_ADDR"
echo "CREDITCARD_CSV=$CREDITCARD_CSV"
echo "PROTOCOL=$PROTOCOL"
echo "ALPHA=$ALPHA"
echo "SEED=$SEED"
echo "ROUNDS=$ROUNDS"
echo "RUN_NAME=$RUN_NAME"

CONFIG="\
  dataset=creditcard \
  model=fraud_mlp \
  partitioning=creditcard_dirichlet \
  partitioning.num_partitions=${NUM_CLIENTS} \
  partitioning.alpha=${ALPHA} \
  algorithm=${PROTOCOL} \
  algorithm.model.last_client_layer=block1 \
  general.seed=${SEED} \
  general.num_rounds=${ROUNDS} \
  client_train_config.lte=1 \
  client_train_config.batch_size=512 \
  optimizer=adam \
  optimizer.lr=0.001 \
  optimizer.weight_decay=0.00001 \
  strategy_config.fraction_fit=1.0 \
  strategy_config.fraction_evaluate=1.0 \
  loss.class_weight=[1.0,50.0] \
  +num_clients=${NUM_CLIENTS} \
  hydra.run.dir=outputs/creditcard/${RUN_NAME} \
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
  export COLEXT_N_CLIENTS=${NUM_CLIENTS}
  export CONFIG='$CONFIG'

  echo SERVER host: \$(hostname)
  echo SERVER CONFIG: \$CONFIG

  echo "===== GPU CHECK ====="
  hostname
  nvidia-smi || true

  python - <<'PY'
  import os
  import torch

  print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
  print("torch version =", torch.__version__)
  print("torch CUDA build =", torch.version.cuda)
  print("torch.cuda.is_available() =", torch.cuda.is_available())

  if torch.cuda.is_available():
      print("GPU count =", torch.cuda.device_count())
      print("GPU name =", torch.cuda.get_device_name(0))
  PY
  echo "====================="

  python -u scripts/py/run_general_server.py \$CONFIG
" > "$ROOT/server_${SLURM_JOB_ID}.log" 2>&1 &

sleep 20

# Clients
for CLIENT_ID in 0 1 2 3; do
  CLIENT_NODE="${HOSTS[$((CLIENT_ID + 1))]}"

  echo "Launching CLIENT${CLIENT_ID} on $CLIENT_NODE"

  srun -N1 -n1 -w "$CLIENT_NODE" --exclusive bash -lc "
    source /data/datasets/$USER/venv_split_algos/bin/activate
    cd '$ROOT'

    export PYTHONPATH='$ROOT'
    export HYDRA_FULL_ERROR=1
    export CREDITCARD_CSV='$CREDITCARD_CSV'
    export COLEXT_SERVER_ADDRESS='${SERVER_ADDR}'
    export COLEXT_CLIENT_ID=${CLIENT_ID}
    export CONFIG='$CONFIG'

    echo CLIENT${CLIENT_ID} host: \$(hostname)
    echo CLIENT${CLIENT_ID} SERVER: \$COLEXT_SERVER_ADDRESS
    echo CLIENT${CLIENT_ID} ID: \$COLEXT_CLIENT_ID

    echo "===== GPU CHECK ====="
    hostname
    nvidia-smi || true

    python - <<'PY'
    import os
    import torch

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch version =", torch.__version__)
    print("torch CUDA build =", torch.version.cuda)
    print("torch.cuda.is_available() =", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU count =", torch.cuda.device_count())
        print("GPU name =", torch.cuda.get_device_name(0))
    PY
    echo "====================="

    python -u scripts/py/run_general_client.py \$CONFIG
  " > "$ROOT/client${CLIENT_ID}_${SLURM_JOB_ID}.log" 2>&1 &
done

wait
echo DONE
