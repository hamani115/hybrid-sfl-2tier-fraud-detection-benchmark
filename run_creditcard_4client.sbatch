#!/bin/bash -l
#SBATCH --job-name=cc_sfl
#SBATCH --partition=compute
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/cc_sfl_%j.out

set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"

source /data/datasets/$USER/venv_split_algos/bin/activate
cd "$ROOT"

mkdir -p "$ROOT/logs/slurm"

export PYTHONPATH="$ROOT"
export HYDRA_FULL_ERROR=1
export CREDITCARD_CSV="/data/datasets/$USER/creditcard/creditcard.csv"

if [ ! -f "$CREDITCARD_CSV" ]; then
  echo "ERROR: creditcard.csv not found at: $CREDITCARD_CSV"
  exit 1
fi

# ------------------------------------------------------------
# Configurable experiment variables
# ------------------------------------------------------------
PROTOCOL="${PROTOCOL:-splitfedv2}"
ALPHA="${ALPHA:-0.5}"
SEED="${SEED:-10}"
ROUNDS="${ROUNDS:-5}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
SPLIT_POINT="${SPLIT_POINT:-block1}"

# Unique port per Slurm job.
# This does not make jobs run in parallel.
#PORT="${PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
PORT=8080

case "$SPLIT_POINT" in
  block1)
    EMBEDDING_DIM=64
    ;;
  block2)
    EMBEDDING_DIM=32
    ;;
  block3)
    EMBEDDING_DIM=16
    ;;
  *)
    echo "ERROR: Unsupported SPLIT_POINT=$SPLIT_POINT"
    echo "Supported values: block1, block2, block3"
    exit 1
    ;;
esac

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
RUN_NAME="${PROTOCOL}_c${NUM_CLIENTS}_alpha${ALPHA}_seed${SEED}_r${ROUNDS}_split${SPLIT_POINT}_run${RUN_ID}"

# All job-specific logs go here
JOB_DIR="$ROOT/logs/creditcard/${RUN_NAME}"
mkdir -p "$JOB_DIR"

# Copy main sbatch stdout/stderr into the job folder too
exec > >(tee "$JOB_DIR/main_${SLURM_JOB_ID}.out") 2>&1

readarray -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SERVER_NODE="${HOSTS[0]}"

REQUIRED_NODES=$((NUM_CLIENTS + 1))

if [ "${#HOSTS[@]}" -lt "$REQUIRED_NODES" ]; then
  echo "ERROR: NUM_CLIENTS=$NUM_CLIENTS requires $REQUIRED_NODES nodes, but only ${#HOSTS[@]} nodes were allocated."
  exit 1
fi

SERVER_ADDR="${SERVER_NODE}:${PORT}"

cat > "$JOB_DIR/run_info.txt" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
RUN_NAME=$RUN_NAME
JOB_DIR=$JOB_DIR
SERVER_NODE=$SERVER_NODE
SERVER_ADDR=$SERVER_ADDR
CREDITCARD_CSV=$CREDITCARD_CSV
PROTOCOL=$PROTOCOL
ALPHA=$ALPHA
SEED=$SEED
ROUNDS=$ROUNDS
NUM_CLIENTS=$NUM_CLIENTS
SPLIT_POINT=$SPLIT_POINT
EMBEDDING_DIM=$EMBEDDING_DIM
HOSTS=${HOSTS[*]}
EOF

echo "SERVER_NODE=$SERVER_NODE"
echo "SERVER_ADDR=$SERVER_ADDR"
echo "CREDITCARD_CSV=$CREDITCARD_CSV"
echo "PROTOCOL=$PROTOCOL"
echo "ALPHA=$ALPHA"
echo "SEED=$SEED"
echo "ROUNDS=$ROUNDS"
echo "NUM_CLIENTS=$NUM_CLIENTS"
echo "RUN_NAME=$RUN_NAME"
echo "SPLIT_POINT=$SPLIT_POINT"
echo "EMBEDDING_DIM=$EMBEDDING_DIM"
echo "JOB_DIR=$JOB_DIR"

CONFIG="\
  dataset=creditcard \
  model=fraud_mlp \
  partitioning=creditcard_dirichlet \
  partitioning.num_partitions=${NUM_CLIENTS} \
  partitioning.alpha=${ALPHA} \
  algorithm=${PROTOCOL} \
  algorithm.model.last_client_layer=${SPLIT_POINT} \
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
  communication.embedding_dim=${EMBEDDING_DIM} \
  communication.bytes_per_float=4 \
  +num_clients=${NUM_CLIENTS} \
  +split_point=${SPLIT_POINT} \
  hydra.run.dir=outputs/creditcard/${RUN_NAME} \
"

export CONFIG

echo "$CONFIG" > "$JOB_DIR/config_overrides.txt"
echo "CONFIG=$CONFIG"

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------
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

  echo '===== GPU CHECK ====='
  hostname
  nvidia-smi || true

  python -c 'import os, torch; print(\"CUDA_VISIBLE_DEVICES =\", os.environ.get(\"CUDA_VISIBLE_DEVICES\")); print(\"torch version =\", torch.__version__); print(\"torch CUDA build =\", torch.version.cuda); print(\"torch.cuda.is_available() =\", torch.cuda.is_available()); print(\"GPU count =\", torch.cuda.device_count() if torch.cuda.is_available() else 0); print(\"GPU name =\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")'

  echo '====================='

  GPU_LOG='$JOB_DIR/gpu_server_${SLURM_JOB_ID}.csv'
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits
      sleep 5
    done
  ) > \$GPU_LOG &
  GPU_MONITOR_PID=\$!

  python -u scripts/py/run_general_server.py \$CONFIG
  STATUS=\$?

  kill \$GPU_MONITOR_PID 2>/dev/null || true
  wait \$GPU_MONITOR_PID 2>/dev/null || true

  exit \$STATUS
" > "$JOB_DIR/server_${SLURM_JOB_ID}.log" 2>&1 &

sleep 20

# ------------------------------------------------------------
# Clients
# ------------------------------------------------------------
for ((CLIENT_ID=0; CLIENT_ID<NUM_CLIENTS; CLIENT_ID++)); do
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

    echo '===== GPU CHECK ====='
    hostname
    nvidia-smi || true

    python -c 'import os, torch; print(\"CUDA_VISIBLE_DEVICES =\", os.environ.get(\"CUDA_VISIBLE_DEVICES\")); print(\"torch version =\", torch.__version__); print(\"torch CUDA build =\", torch.version.cuda); print(\"torch.cuda.is_available() =\", torch.cuda.is_available()); print(\"GPU count =\", torch.cuda.device_count() if torch.cuda.is_available() else 0); print(\"GPU name =\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")'

    echo '====================='

    GPU_LOG='$JOB_DIR/gpu_client${CLIENT_ID}_${SLURM_JOB_ID}.csv'
    (
      while true; do
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw \
          --format=csv,noheader,nounits
        sleep 5
      done
    ) > \$GPU_LOG &
    GPU_MONITOR_PID=\$!

    python -u scripts/py/run_general_client.py \$CONFIG
    STATUS=\$?

    kill \$GPU_MONITOR_PID 2>/dev/null || true
    wait \$GPU_MONITOR_PID 2>/dev/null || true

    exit \$STATUS
  " > "$JOB_DIR/client${CLIENT_ID}_${SLURM_JOB_ID}.log" 2>&1 &
done

wait

echo DONE
echo "Logs saved in: $JOB_DIR"
