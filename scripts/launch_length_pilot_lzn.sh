#!/usr/bin/env bash
# Explicit one-shot launch; no polling or job eviction. Optional physical GPU ID.
set -euo pipefail
gpu_id="${1:-1}"
[[ "$gpu_id" =~ ^[01]$ ]] || { echo 'Expected lzn physical GPU 0 or 1'; exit 2; }
cap="${2:-1024}"
[[ "$cap" =~ ^(512|1024|2048)$ ]] || { echo 'Expected cap 512, 1024 or 2048'; exit 2; }
max_util="${PILOT_MAX_GPU_UTILIZATION:-20}"
memory_fraction="${PILOT_GPU_MEMORY_UTILIZATION:-0.40}"
[[ "$max_util" =~ ^[0-9]+$ ]] && (( max_util <= 100 )) || exit 2
cd /home/lzn/dprandopt_run
out="tables/length_pilot_32_max${cap}"
mkdir -p logs "$out"
read -r free_mb util < <(nvidia-smi -i "$gpu_id" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
if (( free_mb < 24576 || util > max_util )); then
  echo "NOT STARTED: GPU ${gpu_id} free=${free_mb}MiB utilization=${util}%; need >=24576MiB and <=${max_util}%."
  exit 75
fi
export CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1
export PYTHONPATH=/home/lzn/dprandopt_run:/home/lzn/dprandopt_run/RandOpt
export RANDOPT_BASE_WEIGHTS_DEVICE=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/lkx/miniconda3/envs/opcd/bin/python
"$py" scripts/build_length_pilot.py \
  --reference tables/lzn_gpu1_route1_3b_n1000_seed20261013_max512.npz \
  --out-dir "$out" --max-tokens "$cap" --prepare-only
# Hold the lock for the wrapper lifetime; never run two writers for one pilot.
exec 9>tables/length_pilot.run.lock
flock -n 9 || { echo 'NOT STARTED: pilot already locked'; exit 75; }
log="logs/length_pilot_32_max${cap}_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" scripts/run_resumable.py --max-retries 3 --sleep-seconds 60 -- \
  "$py" scripts/build_length_pilot.py \
  --reference tables/lzn_gpu1_route1_3b_n1000_seed20261013_max512.npz \
  --out-dir "$out" --count 32 --max-tokens "$cap" \
  --gpu-memory-utilization "$memory_fraction" > "$log" 2>&1 < /dev/null &
echo "STARTED wrapper_pid=$! log=$log"
