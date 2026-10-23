#!/usr/bin/env bash
set -euo pipefail
cd /home/lzn/dprandopt_run
gpu_id="${1:-0}"
[[ "$gpu_id" =~ ^[01]$ ]] || exit 2
read -r free_mb util < <(nvidia-smi -i "$gpu_id" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
if (( free_mb < 38000 || util > 20 )); then
  echo "NOT STARTED: GPU $gpu_id free=${free_mb}MiB utilization=${util}%"; exit 75
fi
mkdir -p logs
exec 9>tables/length_pilot.run.lock
flock -n 9 || { echo 'NOT STARTED: another pilot owns the lock'; exit 75; }
export CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1
export PYTHONPATH=/home/lzn/dprandopt_run:/home/lzn/dprandopt_run/RandOpt
export RANDOPT_BASE_WEIGHTS_DEVICE=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/lkx/miniconda3/envs/opcd/bin/python
out=tables/batched_controls_20261023
"$py" scripts/run_batched_controls.py --reference tables/lzn_gpu1_route1_3b_n1000_seed20261013_max512.npz \
  --production-pilot tables/length_pilot_32_max1024 --out-dir "$out" --prepare-only
log="logs/batched_controls_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" scripts/run_batched_controls.py --reference tables/lzn_gpu1_route1_3b_n1000_seed20261013_max512.npz \
  --production-pilot tables/length_pilot_32_max1024 --out-dir "$out" --gpu-memory-utilization .65 \
  > "$log" 2>&1 < /dev/null &
echo "STARTED pid=$! log=$log"
