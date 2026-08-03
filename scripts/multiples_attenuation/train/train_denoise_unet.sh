#!/usr/bin/env bash
# Run multiple seeds for the fixed multiples paired SEG-Y dataset.
# Output dirs don't collide: experiment name is ``<yaml name>_seed<seed>``.
# Configure N_SEEDS / START_SEED below.

set -euo pipefail

# ---------- Configuration ----------
CUDA_VISIBLE_DEVICES="3,4" # physical GPUs, comma-separated
NPROC_PER_NODE=2         # must match the number of visible GPUs
N_SEEDS=3                  # number of seeds
START_SEED=42              # first seed; subsequent seeds are START_SEED+1, START_SEED+2, ...
MASTER_PORT=28500          # base port for torchrun; incremented per run to avoid EADDRINUSE
TORCHRUN_EXTRA=""          # optional: extra flags for torchrun, e.g. "--standalone"
# ------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/multiples_attenuation/denoise_unet.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/multiples_attenuation/train/train_denoise_unet.py"

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

n_total=${N_SEEDS}
run_idx=0

for ((s = 0; s < N_SEEDS; s++)); do
  seed=$((START_SEED + s))
  run_idx=$((run_idx + 1))
  run_name="${NAME_BASE}_seed${seed}"
  sed -E \
    -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
    -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name}"'/' \
    "${BASE_CONFIG}" >"${tmpcfg}"
  port=$((MASTER_PORT + run_idx - 1))
  echo "[$(date -Iseconds)] (${run_idx}/${n_total}) seed=${seed} name=${run_name} port=${port}"
  cd "${REPO_ROOT}"
  # shellcheck disable=SC2086
  torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" "${PY_SCRIPT}" --config "${tmpcfg}"
done

echo "[$(date -Iseconds)] Done ${n_total} run(s) (${N_SEEDS} seed(s))."
