#!/usr/bin/env bash
set -uo pipefail

ROUNDS=5
POLL_SECONDS=60

SEEDS=(10)
ALPHAS=(0.1)
CLIENTS=(2 3 4)
SPLITS=(block1)
PROTOCOLS=(splitfedv1 splitfedv2)

echo "Severe non-IID stress test: alpha=0.1, rounds=$ROUNDS"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for NUM_CLIENTS_VALUE in "${CLIENTS[@]}"; do
      for SPLIT in "${SPLITS[@]}"; do
        for PROTOCOL in "${PROTOCOLS[@]}"; do
          NODES=$((NUM_CLIENTS_VALUE + 1))

          echo ""
          echo "Submitting stress test: PROTOCOL=$PROTOCOL CLIENTS=$NUM_CLIENTS_VALUE ALPHA=$ALPHA SEED=$SEED SPLIT=$SPLIT ROUNDS=$ROUNDS"

          JOB_ID=$(
            NUM_CLIENTS="$NUM_CLIENTS_VALUE" \
            SPLIT_POINT="$SPLIT" \
            PROTOCOL="$PROTOCOL" \
            ALPHA="$ALPHA" \
            SEED="$SEED" \
            ROUNDS="$ROUNDS" \
            sbatch --parsable \
              --nodes="$NODES" \
              --ntasks="$NODES" \
              --job-name="cc_sfl_stress_${NUM_CLIENTS_VALUE}c" \
              --output="cc_sfl_stress_${NUM_CLIENTS_VALUE}c_%j.out" \
              run_creditcard_4client.sbatch
          )

          echo "Submitted job: $JOB_ID"

          while squeue -j "$JOB_ID" -h | grep -q .; do
            echo "Job $JOB_ID still running/pending... $(date)"
            sleep "$POLL_SECONDS"
          done

          sleep 10
          sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
        done
      done
    done
  done
done
