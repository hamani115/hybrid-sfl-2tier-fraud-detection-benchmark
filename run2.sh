#!/usr/bin/env bash
set -euo pipefail

SEEDS=(10 30)
ALPHAS=(0.5 0.1 100)
SPLITS=(block1 block2 block3)
PROTOCOLS=(splitfedv1 splitfedv2)
ROUNDS=50

POLL_SECONDS=60
JOB_LOG="submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitted jobs will be saved in: $JOB_LOG"
echo "Start time: $(date)"
echo

submit_and_wait() {
  local seed="$1"
  local alpha="$2"
  local split="$3"
  local protocol="$4"

  printf "\n============================================================\n"
  printf "Submitting: PROTOCOL=%s ALPHA=%s SEED=%s SPLIT_POINT=%s ROUNDS=%s\n" \
    "$protocol" "$alpha" "$seed" "$split" "$ROUNDS"
  printf "============================================================\n"

  JOB_ID=$(
    SPLIT_POINT="$split" \
    PROTOCOL="$protocol" \
    ALPHA="$alpha" \
    SEED="$seed" \
    ROUNDS="$ROUNDS" \
    sbatch --parsable run_creditcard_4client.sbatch
  )

  echo "Submitted job: $JOB_ID"
  echo "$JOB_ID,$protocol,$alpha,$seed,$split,$ROUNDS" >> "$JOB_LOG"

  while squeue -j "$JOB_ID" -h | grep -q "$JOB_ID"; do
    echo "Job $JOB_ID still running/pending... $(date)"
    squeue -j "$JOB_ID"
    sleep "$POLL_SECONDS"
  done

  echo "Job $JOB_ID left the queue. Checking final state..."

  # Give Slurm accounting a few seconds to update.
  sleep 10

  sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList

  STATE=$(sacct -X -j "$JOB_ID" --format=State --noheader | awk 'NR==1 {print $1}')

  if [[ "$STATE" != "COMPLETED" ]]; then
    echo "ERROR: Job $JOB_ID finished with state: $STATE"
    echo "Check logs:"
    echo "  cc_sfl_4c_${JOB_ID}.out"
    echo "  server_${JOB_ID}.log"
    echo "  client0_${JOB_ID}.log"
    echo "  client1_${JOB_ID}.log"
    echo "  client2_${JOB_ID}.log"
    echo "  client3_${JOB_ID}.log"
    exit 1
  fi

  echo "Job $JOB_ID completed successfully."
}

for SEED in "${SEEDS[@]}"; do
  printf "\n==================== SEED = %s ====================\n" "$SEED"

  for ALPHA in "${ALPHAS[@]}"; do
    printf "\n---------- ALPHA = %s ----------\n" "$ALPHA"

    for SPLIT in "${SPLITS[@]}"; do
      printf "\n----- SPLIT = %s -----\n" "$SPLIT"

      for PROTOCOL in "${PROTOCOLS[@]}"; do
        submit_and_wait "$SEED" "$ALPHA" "$SPLIT" "$PROTOCOL"
      done
    done
  done
done

printf "\n============== ALL JOBS DONE ==============\n"
echo "End time: $(date)"
