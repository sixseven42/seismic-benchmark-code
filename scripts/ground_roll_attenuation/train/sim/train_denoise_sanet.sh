#!/usr/bin/env bash
set -euo pipefail

# ---------- Configuration ----------
CUDA_VISIBLE_DEVICES="6,7" 
NPROC_PER_NODE=2
NOISE_LEVELS=(1.0)
N_SEEDS=1
START_SEED=42
MASTER_PORT=29900
TORCHRUN_EXTRA=""
# ------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/ground_roll_attenuation/sim/denoise_sanet.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/ground_roll_attenuation/train/train_denoise_sanet.py"

export CUDA_VISIBLE_DEVICES

if [[ ! -f "${BASE_CONFIG}" ]]; then echo "Config not found: ${BASE_CONFIG}" >&2; exit 1; fi
if [[ ! -f "${PY_SCRIPT}" ]]; then echo "Script not found: ${PY_SCRIPT}" >&2; exit 1; fi

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${BASE_CONFIG}" | sed -E 's/^[[:space:]]*name:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
tmpcfg="$(mktemp)"
cleanup() { rm -f "${tmpcfg}"; }
trap cleanup EXIT

n_levels=${#NOISE_LEVELS[@]}
n_total=$((n_levels * N_SEEDS))
run_idx=0

for level in "${NOISE_LEVELS[@]}"; do
  for ((s = 0; s < N_SEEDS; s++)); do
    seed=$((START_SEED + s))
    run_idx=$((run_idx + 1))
    run_name="${NAME_BASE}_level${level}_seed${seed}"
    sed -E \
      -e '/input_path:/s/(noisy_)[0-9.]+(\.sgy)/\1'"${level}"'\2/' \
      -e '/target_path:/s/(noise_)[0-9.]+(\.sgy)/\1'"${level}"'\2/' \
      -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
      -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name}"'/' \
      "${BASE_CONFIG}" >"${tmpcfg}"
    port=$((MASTER_PORT + run_idx - 1))
    echo "[$(date -Iseconds)] (${run_idx}/${n_total}) level=${level} seed=${seed} name=${run_name} port=${port}"
    cd "${REPO_ROOT}"
    torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" "${PY_SCRIPT}" --config "${tmpcfg}"
  done
done

echo "[$(date -Iseconds)] Done ${n_total} runs (${n_levels} level(s) x ${N_SEEDS} seed(s))."
