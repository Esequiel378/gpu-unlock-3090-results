#!/usr/bin/env bash
# Runs bench.py as root with the vLLM venv. On NixOS, set COMFY_UNIT to a
# ComfyUI unit that already carries the loader workarounds (LD_LIBRARY_PATH,
# TRITON_*, CC); they are copied from its live environment so store paths
# stay current. Elsewhere those variables are simply left unset.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${BENCH_DIR:?set BENCH_DIR to the bench state directory}"
: "${COMFY_DIR:?set COMFY_DIR to the ComfyUI install}"
export BENCH_DIR COMFY_DIR
venv=${VLLM_VENV:-$BENCH_DIR/vllm-venv}

unit_env() { systemctl show "${COMFY_UNIT:-comfyui}" -p Environment --value | tr ' ' '\n' | sed -n "s/^$1=//p"; }

for v in LD_LIBRARY_PATH TRITON_LIBCUDA_PATH CC; do
  val=$(unit_env "$v")
  if [ -n "$val" ]; then export "$v=$val"; fi
done
ptxas_wrapper=$(unit_env TRITON_PTXAS_PATH)
if [ -n "$ptxas_wrapper" ]; then
  # Triton's bundled ptxas is a generic-linux binary: run it through the NixOS loader
  loader=$(awk '/^exec/ {print $2}' "$ptxas_wrapper")
  ptxas=$(ls "$venv"/lib/python3*/site-packages/triton/backends/nvidia/bin/ptxas)
  printf '#!/bin/sh\nexec %s %s "$@"\n' "$loader" "$ptxas" > "$BENCH_DIR/ptxas"
  chmod +x "$BENCH_DIR/ptxas"
  export TRITON_PTXAS_PATH=$BENCH_DIR/ptxas
fi

export HOME=$BENCH_DIR
export XDG_CACHE_HOME=$BENCH_DIR/.cache
export VLLM_CACHE_ROOT=$BENCH_DIR/.cache/vllm
export HF_HUB_OFFLINE=1
export UV_PYTHON_DOWNLOADS=never
# FlashInfer would JIT its sampler with nvcc, which the test machine didn't have
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=$venv/bin:$PATH

exec "$venv/bin/python" "$here/bench.py" "$@"
