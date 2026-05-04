#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Hybrid SFL Credit Card Fraud Benchmark Runner
#
# Runs sequentially:
#   1. Main benchmark
#   2. Client-count ablation
#
# If one job fails, the script prints an error to stderr,
# records the failure, and continues to the next run.
# ============================================================

ROUNDS=50
POLL_SECONDS=60

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_LOG="submitted_jobs_${TIMESTAMP}.csv"
FAILED_LOG="failed_jobs_${TIMESTAMP}.csv"

# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------
info() {
  echo "$@"
}

err() {
  echo "$@" >&2
}

# ------------------------------------------------------------
# Create CSV logs
# ------------------------------------------------------------
echo "timestamp,benchmark,job_id,protocol,num_clients,alpha,seed,split_point,rounds,state,exit_code,elapsed,slurm_out,server_log,client_logs" > "$JOB_LOG"
echo "timestamp,benchmark,job_id,protocol,num_clients,alpha,seed,split_point,rounds,state,exit_code,elapsed,reason" > "$FAILED_LOG"

info "Submitted jobs will be saved in: $JOB_LOG"
info "Failed jobs will be saved in: $FAILED_LOG"
info "Start time: $(date)"
info ""

# ------------------------------------------------------------
# Submit one job and wait for it to finish
# ------------------------------------------------------------
submit_and_wait() {
  local benchmark="$1"
  local protocol="$2"
  local num_clients="$3"
  local alpha="$4"
  local seed="$5"
  local split_point="$6"

  local nodes=$((num_clients + 1))
  local job_name="cc_sfl_${num_clients}c"
  local slurm_out="cc_sfl_${num_clients}c_%j.out"

  info ""
  info "============================================================"
  info "Benchmark:   $benchmark"
  info "Protocol:    $protocol"
  info "Clients:     $num_clients"
  info "Alpha:       $alpha"
  info "Seed:        $seed"
  info "Split point: $split_point"
  info "Rounds:      $ROUNDS"
  info "Nodes/tasks: $nodes"
  info "============================================================"

  JOB_ID=$(
    NUM_CLIENTS="$num_clients" \
    SPLIT_POINT="$split_point" \
    PROTOCOL="$protocol" \
    ALPHA="$alpha" \
    SEED="$seed" \
    ROUNDS="$ROUNDS" \
    sbatch --parsable \
      --nodes="$nodes" \
      --ntasks="$nodes" \
      --job-name="$job_name" \
      --output="$slurm_out" \
      run_creditcard_4client.sbatch
  )

  if [[ -z "${JOB_ID:-}" ]]; then
    err "ERROR: Failed to submit job for benchmark=$benchmark protocol=$protocol clients=$num_clients alpha=$alpha seed=$seed split=$split_point"
    echo "$(date),$benchmark,NA,$protocol,$num_clients,$alpha,$seed,$split_point,$ROUNDS,SUBMIT_FAILED,NA,NA,sbatch submission returned empty JOB_ID" >> "$FAILED_LOG"
    return 1
  fi

  info "Submitted job: $JOB_ID"

  local client_logs=""
  for ((i=0; i<num_clients; i++)); do
    client_logs="${client_logs} client${i}_${JOB_ID}.log"
  done

  echo "$(date),$benchmark,$JOB_ID,$protocol,$num_clients,$alpha,$seed,$split_point,$ROUNDS,SUBMITTED,NA,NA,cc_sfl_${num_clients}c_${JOB_ID}.out,server_${JOB_ID}.log,\"$client_logs\"" >> "$JOB_LOG"

  while squeue -j "$JOB_ID" -h | grep -q .; do
    info "Job $JOB_ID still running/pending... $(date)"
    squeue -j "$JOB_ID"
    sleep "$POLL_SECONDS"
  done

  info "Job $JOB_ID left the queue. Checking final state..."
  sleep 10

  sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList

  local state
  local exit_code
  local elapsed

  state=$(sacct -X -j "$JOB_ID" --format=State --noheader | awk 'NR==1 {print $1}')
  exit_code=$(sacct -X -j "$JOB_ID" --format=ExitCode --noheader | awk 'NR==1 {print $1}')
  elapsed=$(sacct -X -j "$JOB_ID" --format=Elapsed --noheader | awk 'NR==1 {print $1}')

  if [[ -z "$state" ]]; then
    state="UNKNOWN"
  fi

  if [[ -z "$exit_code" ]]; then
    exit_code="UNKNOWN"
  fi

  if [[ -z "$elapsed" ]]; then
    elapsed="UNKNOWN"
  fi

  echo "$(date),$benchmark,$JOB_ID,$protocol,$num_clients,$alpha,$seed,$split_point,$ROUNDS,$state,$exit_code,$elapsed,cc_sfl_${num_clients}c_${JOB_ID}.out,server_${JOB_ID}.log,\"$client_logs\"" >> "$JOB_LOG"

  if [[ "$state" != "COMPLETED" ]]; then
    err ""
    err "ERROR: Job $JOB_ID failed or did not complete."
    err "Benchmark:   $benchmark"
    err "Protocol:    $protocol"
    err "Clients:     $num_clients"
    err "Alpha:       $alpha"
    err "Seed:        $seed"
    err "Split point: $split_point"
    err "State:       $state"
    err "Exit code:   $exit_code"
    err "Elapsed:     $elapsed"
    err "Check logs:"
    err "  cc_sfl_${num_clients}c_${JOB_ID}.out"
    err "  server_${JOB_ID}.log"
    for ((i=0; i<num_clients; i++)); do
      err "  client${i}_${JOB_ID}.log"
    done
    err ""

    echo "$(date),$benchmark,$JOB_ID,$protocol,$num_clients,$alpha,$seed,$split_point,$ROUNDS,$state,$exit_code,$elapsed,job did not complete" >> "$FAILED_LOG"

    # Continue instead of exiting.
    return 1
  fi

  info "Job $JOB_ID completed successfully."
  return 0
}

# ============================================================
# Benchmark 1: Main benchmark
# ============================================================

MAIN_SEEDS=(10 30)
MAIN_ALPHAS=(100 1.0 0.5)
MAIN_SPLITS=(block1 block2 block3)
MAIN_PROTOCOLS=(splitfedv1 splitfedv2)
MAIN_CLIENTS=4

info ""
info "############################################################"
info "# BENCHMARK 1: MAIN BENCHMARK"
info "############################################################"

for SEED in "${MAIN_SEEDS[@]}"; do
  for ALPHA in "${MAIN_ALPHAS[@]}"; do
    for SPLIT in "${MAIN_SPLITS[@]}"; do
      for PROTOCOL in "${MAIN_PROTOCOLS[@]}"; do
        submit_and_wait \
          "main_benchmark" \
          "$PROTOCOL" \
          "$MAIN_CLIENTS" \
          "$ALPHA" \
          "$SEED" \
          "$SPLIT"
      done
    done
  done
done

# ============================================================
# Benchmark 2: Client-count ablation
# ============================================================

CLIENT_ABLATION_SEEDS=(10 30)
CLIENT_ABLATION_ALPHAS=(0.5)
CLIENT_ABLATION_SPLITS=(block1)
CLIENT_ABLATION_PROTOCOLS=(splitfedv1 splitfedv2)
CLIENT_ABLATION_CLIENTS=(2 3 4)

info ""
info "############################################################"
info "# BENCHMARK 2: CLIENT-COUNT ABLATION"
info "############################################################"

for SEED in "${CLIENT_ABLATION_SEEDS[@]}"; do
  for ALPHA in "${CLIENT_ABLATION_ALPHAS[@]}"; do
    for NUM_CLIENTS_VALUE in "${CLIENT_ABLATION_CLIENTS[@]}"; do
      for SPLIT in "${CLIENT_ABLATION_SPLITS[@]}"; do
        for PROTOCOL in "${CLIENT_ABLATION_PROTOCOLS[@]}"; do
          submit_and_wait \
            "client_count_ablation" \
            "$PROTOCOL" \
            "$NUM_CLIENTS_VALUE" \
            "$ALPHA" \
            "$SEED" \
            "$SPLIT"
        done
      done
    done
  done
done

info ""
info "============== ALL REQUESTED JOBS FINISHED =============="
info "End time: $(date)"
info "Job log: $JOB_LOG"
info "Failed job log: $FAILED_LOG"

if [[ -s "$FAILED_LOG" ]]; then
  err ""
  err "Some jobs may have failed. Check:"
  err "  $FAILED_LOG"
  err ""
fi
