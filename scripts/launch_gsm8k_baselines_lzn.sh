#!/usr/bin/env bash
set -euo pipefail
cd /home/lzn/dprandopt_run
gpu_id="${1:-0}"
[[ "$gpu_id" =~ ^[01]$ ]] || exit 2
read -r free_mb util < <(nvidia-smi -i "$gpu_id" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
if (( free_mb < 38000 || util > 20 )); then
  echo "NOT STARTED: GPU $gpu_id free=${free_mb}MiB utilization=${util}%"; exit 75
fi
exec 9>tables/length_pilot.run.lock
flock -n 9 || exit 75
export CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1
export PYTHONPATH=/home/lzn/dprandopt_run:/home/lzn/dprandopt_run/RandOpt
py=/home/lkx/miniconda3/envs/opcd/bin/python
out=tables/gsm8k_baselines_20261023
"$py" scripts/run_gsm8k_baselines.py --data-dir data/gsm8k --out-dir "$out" \
  --model /home/lkx/OPCD/models/Qwen2.5-3B-Instruct --prepare-only
log="logs/gsm8k_baselines_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" scripts/run_gsm8k_baselines.py --data-dir data/gsm8k --out-dir "$out" \
  --model /home/lkx/OPCD/models/Qwen2.5-3B-Instruct --gpu-memory-utilization .65 \
  > "$log" 2>&1 < /dev/null &
echo "STARTED pid=$! log=$log"
