#!/usr/bin/env bash
# Nested loop: for each noise-intensity level, run multiple seeds.
# Output dirs: ``<yaml name>_level<level>_seed<seed>``.
# Configure NOISE_LEVELS / N_SEEDS / START_SEED below.

set -euo pipefail

# ---------- Configuration ----------
CUDA_VISIBLE_DEVICES="4,5,6,7" 
NPROC_PER_NODE=4
NOISE_LEVELS=(1.0 3.0 5.0 7.0 9.0)
N_SEEDS=3
START_SEED=42
MASTER_PORT=29800
TORCHRUN_EXTRA=""
# ------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/ground_roll_attenuation/denoise_enhanced_unet.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/ground_roll_attenuation/train/train_denoise_enhanced_unet.py"

export CUDA_VISIBLE_DEVICES

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config not found: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${BASE_CONFIG}" | sed -E 's/^[[:space:]]*name:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
if [[ -z "${NAME_BASE}" ]]; then
  echo "Could not parse experiment.name from ${BASE_CONFIG}" >&2
  exit 1
fi

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
    # shellcheck disable=SC2086
    torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" "${PY_SCRIPT}" --config "${tmpcfg}"
  done
done

echo "[$(date -Iseconds)] Done ${n_total} runs (${n_levels} level(s) x ${N_SEEDS} seed(s))."
