#!/usr/bin/env bash
# Recursively download selected experiment directories over SFTP with resume.
# Edit the defaults below or override them with environment variables, e.g.:
#   SFTP_HOST=192.0.2.10 SFTP_PORT=22 bash scripts/ground_roll_attenuation/download_results_0822.sh

set -euo pipefail

SFTP_HOST="${SFTP_HOST:-117.50.174.129}"
SFTP_PORT="${SFTP_PORT:-23}"
SFTP_USER="${SFTP_USER:-root}"
REMOTE_BASE="${REMOTE_BASE:-/cloud/cloud-s3fs/ground_roll/results_0822}"
LOCAL_DIR="${LOCAL_DIR:-/data/shared/benchmark/ground_roll/results_0822}"

EXPERIMENTS=(
  "denoise_atten_unet_base0822_level1.0_seed42"
  "denoise_atten_unet_base0822_level1.0_seed43"
  "denoise_atten_unet_base0822_level1.0_seed44"
  "denoise_unet_base0822_level1.0_seed42"
  "denoise_unet_base0822_level1.0_seed43"
  "denoise_unet_base0822_level1.0_seed44"

)

if [[ -z "${SFTP_PASSWORD:-}" ]]; then
  read -r -s -p "Password for ${SFTP_USER}@${SFTP_HOST}: " SFTP_PASSWORD
  printf '\n'
fi
if [[ -z "${SFTP_PASSWORD}" ]]; then
  echo "Password must not be empty." >&2
  exit 1
fi
export SFTP_PASSWORD

mkdir -p "${LOCAL_DIR}"
echo "Downloading ${#EXPERIMENTS[@]} experiment(s) to ${LOCAL_DIR}"
export SFTP_HOST SFTP_PORT SFTP_USER REMOTE_BASE LOCAL_DIR
export SFTP_EXPERIMENTS="$(printf '%s\n' "${EXPERIMENTS[@]}")"

# SFTP batch mode disables password prompts. This PTY wrapper keeps SFTP
# interactive while issuing the commands automatically.
python3 - <<'PY'
import os
import pty
import select
import subprocess
import sys
import time
import fcntl
import termios


def read_until(master_fd, process, marker, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    tail = b""
    while marker not in tail:
        if process.poll() is not None:
            raise RuntimeError(f"sftp exited with status {process.returncode}")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {marker!r}")
        ready, _, _ = select.select([master_fd], [], [], 1.0)
        if not ready:
            continue
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            chunk = b""
        if not chunk:
            continue
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        tail = (tail + chunk)[-8192:]


def setup_tty():
    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


host = os.environ["SFTP_HOST"]
port = os.environ["SFTP_PORT"]
user = os.environ["SFTP_USER"]
remote_base = os.environ["REMOTE_BASE"].rstrip("/")
local_dir = os.environ["LOCAL_DIR"]
password = os.environ["SFTP_PASSWORD"].encode() + b"\n"
experiments = [x for x in os.environ["SFTP_EXPERIMENTS"].splitlines() if x]

master, slave = pty.openpty()
command = [
    "sftp", "-P", port,
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=10",
    f"{user}@{host}",
]
process = subprocess.Popen(
    command,
    stdin=slave,
    stdout=slave,
    stderr=slave,
    cwd=local_dir,
    close_fds=True,
    preexec_fn=setup_tty,
)
os.close(slave)

try:
    read_until(master, process, b"password:", timeout=60)
    os.write(master, password)
    read_until(master, process, b"sftp> ", timeout=60)
    for experiment in experiments:
        command_line = f"get -a -r {remote_base}/{experiment}\n".encode()
        os.write(master, command_line)
        read_until(master, process, b"sftp> ")
    os.write(master, b"bye\n")
    if process.wait() != 0:
        raise RuntimeError(f"sftp exited with status {process.returncode}")
finally:
    os.close(master)
PY
unset SFTP_PASSWORD

echo "Download complete. Re-run this script after an interruption to resume."
