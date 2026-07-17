#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  rollout_success_rate.sh \
    --policy POLICY \
    (--env ENV | --env-list "ENV1 ENV2 ...") \
    --checkpoint CHECKPOINT \
    --world-idx-list "0 1 2 3 4" \
    [--num-rollouts 1] \
    [--duration 30] \
    [--python python] \
    [--args-file-rollout path/to/args.txt] \
    [--output-dir path/to/output]

Description:
  Run Rollout.py sequentially for each env and each world_idx, and repeat each by --num-rollouts.
  It parses "Rollout result: success|failure" and reports per-world and overall success rate.
EOF
}

POLICY=""
ENV_NAME=""
ENV_LIST_STR=""
CHECKPOINT=""
WORLD_IDX_LIST_STR=""
NUM_ROLLOUTS=1
DURATION=30
PYTHON_BIN="python"
ARGS_FILE_ROLLOUT=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      POLICY="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --env-list)
      ENV_LIST_STR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --world-idx-list)
      WORLD_IDX_LIST_STR="$2"
      shift 2
      ;;
    --num-rollouts)
      NUM_ROLLOUTS="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --args-file-rollout)
      ARGS_FILE_ROLLOUT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$POLICY" || -z "$CHECKPOINT" || -z "$WORLD_IDX_LIST_STR" ]]; then
  echo "[ERROR] --policy, --checkpoint, --world-idx-list are required." >&2
  usage
  exit 1
fi

if [[ -z "$ENV_NAME" && -z "$ENV_LIST_STR" ]]; then
  echo "[ERROR] Specify either --env or --env-list." >&2
  exit 1
fi

if [[ -n "$ENV_NAME" && -n "$ENV_LIST_STR" ]]; then
  echo "[ERROR] Use only one of --env or --env-list." >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if ! [[ "$NUM_ROLLOUTS" =~ ^[0-9]+$ ]] || [[ "$NUM_ROLLOUTS" -le 0 ]]; then
  echo "[ERROR] --num-rollouts must be a positive integer: $NUM_ROLLOUTS" >&2
  exit 1
fi

if [[ -n "$ARGS_FILE_ROLLOUT" ]] && [[ ! -f "$ARGS_FILE_ROLLOUT" ]]; then
  echo "[ERROR] args file not found: $ARGS_FILE_ROLLOUT" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROLLOUT_PY="$REPO_ROOT/robo_manip_baselines/bin/Rollout.py"

if [[ ! -f "$ROLLOUT_PY" ]]; then
  echo "[ERROR] Rollout.py not found: $ROLLOUT_PY" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ -n "$ENV_NAME" ]]; then
    output_env_tag="$ENV_NAME"
  else
    output_env_tag="multi_env"
  fi
  OUTPUT_DIR="$REPO_ROOT/robo_manip_baselines/misc/result_rollout/$(date +%Y%m%d_%H%M%S)_${POLICY}_${output_env_tag}"
fi
mkdir -p "$OUTPUT_DIR"

IFS=' ' read -r -a WORLD_IDX_LIST <<< "$WORLD_IDX_LIST_STR"
if [[ "${#WORLD_IDX_LIST[@]}" -eq 0 ]]; then
  echo "[ERROR] --world-idx-list is empty" >&2
  exit 1
fi

if [[ -n "$ENV_NAME" ]]; then
  ENV_LIST=("$ENV_NAME")
else
  IFS=' ' read -r -a ENV_LIST <<< "$ENV_LIST_STR"
fi
if [[ "${#ENV_LIST[@]}" -eq 0 ]]; then
  echo "[ERROR] env list is empty" >&2
  exit 1
fi

declare -A SUCCESS_BY_ENV_WORLD
declare -A TOTAL_BY_ENV_WORLD
for env in "${ENV_LIST[@]}"; do
  for world_idx in "${WORLD_IDX_LIST[@]}"; do
    key="${env}|${world_idx}"
    SUCCESS_BY_ENV_WORLD["$key"]=0
    TOTAL_BY_ENV_WORLD["$key"]=0
  done
done

TOTAL_SUCCESS=0
TOTAL_TRIALS=0
LOG_FILE="$OUTPUT_DIR/rollout_success_rate.log"

{
  echo "[INFO] Start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[INFO] policy=$POLICY env_list=${ENV_LIST[*]}"
  echo "[INFO] checkpoint=$CHECKPOINT"
  echo "[INFO] world_idx_list=${WORLD_IDX_LIST[*]}"
  echo "[INFO] num_rollouts=$NUM_ROLLOUTS duration=$DURATION"
  echo "[INFO] output_dir=$OUTPUT_DIR"

  for env in "${ENV_LIST[@]}"; do
    echo "[INFO] ===== env=${env} ====="

    for ((rep=1; rep<=NUM_ROLLOUTS; rep++)); do
      echo "[INFO] ===== repeat ${rep}/${NUM_ROLLOUTS} ====="

      for world_idx in "${WORLD_IDX_LIST[@]}"; do
        echo "[INFO] --- world_idx=${world_idx} ---"

        cmd=(
          "$PYTHON_BIN" "$ROLLOUT_PY"
          "$POLICY" "$env"
          --checkpoint "$CHECKPOINT"
          --duration "$DURATION"
          --no_plot
          --no_render
          --save_last_image
          --output_image_dir "$OUTPUT_DIR"
          --world_idx "$world_idx"
        )

        if [[ -n "$ARGS_FILE_ROLLOUT" ]]; then
          cmd+=("@$ARGS_FILE_ROLLOUT")
        fi

        set +e
        cmd_output="$("${cmd[@]}" 2>&1)"
        cmd_status=$?
        set -e

        echo "$cmd_output"

        if [[ "$cmd_status" -ne 0 ]]; then
          echo "[WARN] Rollout command failed (env=$env, world_idx=$world_idx, repeat=$rep, exit=$cmd_status)"
        fi

        result_line="$(printf '%s\n' "$cmd_output" | grep -E '^Rollout result: (success|failure)$' | tail -n 1 || true)"
        result="${result_line#Rollout result: }"

        if [[ "$result" != "success" && "$result" != "failure" ]]; then
          echo "[WARN] Could not parse rollout result for env=$env world_idx=$world_idx repeat=$rep"
          result="failure"
        fi

        key="${env}|${world_idx}"
        TOTAL_BY_ENV_WORLD["$key"]=$(( TOTAL_BY_ENV_WORLD["$key"] + 1 ))
        TOTAL_TRIALS=$(( TOTAL_TRIALS + 1 ))

        if [[ "$result" == "success" ]]; then
          SUCCESS_BY_ENV_WORLD["$key"]=$(( SUCCESS_BY_ENV_WORLD["$key"] + 1 ))
          TOTAL_SUCCESS=$(( TOTAL_SUCCESS + 1 ))
        fi

        echo "[INFO] Parsed result: $result"
      done
    done
  done

  echo ""
  echo "[SUMMARY] ===== per env/world_idx ====="
  for env in "${ENV_LIST[@]}"; do
    for world_idx in "${WORLD_IDX_LIST[@]}"; do
      key="${env}|${world_idx}"
      succ="${SUCCESS_BY_ENV_WORLD[$key]}"
      tot="${TOTAL_BY_ENV_WORLD[$key]}"
      rate="$(awk -v s="$succ" -v t="$tot" 'BEGIN { if (t==0) print "0.00"; else printf "%.2f", (100.0*s)/t }')"
      echo "[SUMMARY] env=${env} world_idx=${world_idx}: ${succ}/${tot} (${rate}%)"
    done
  done

  total_rate="$(awk -v s="$TOTAL_SUCCESS" -v t="$TOTAL_TRIALS" 'BEGIN { if (t==0) print "0.00"; else printf "%.2f", (100.0*s)/t }')"
  echo "[SUMMARY] ===== overall ====="
  echo "[SUMMARY] success=${TOTAL_SUCCESS}/${TOTAL_TRIALS} (${total_rate}%)"
  echo "[INFO] End: $(date '+%Y-%m-%d %H:%M:%S')"
} | tee "$LOG_FILE"

echo "[DONE] log: $LOG_FILE"
