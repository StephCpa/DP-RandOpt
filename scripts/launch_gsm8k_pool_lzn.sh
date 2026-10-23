#!/usr/bin/env bash
# First frozen GSM8K pool; bounded retries resume the same checkpoint.
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
out=tables/gsm8k_n1001_seed20261024_items20261023_max1024.npz
[[ ! -e "$out" ]] || { echo 'Already complete'; exit 0; }
export CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1
export PYTHONPATH=/home/lzn/dprandopt_run:/home/lzn/dprandopt_run/RandOpt
export RANDOPT_BASE_WEIGHTS_DEVICE=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RANDOPT_MAX_MODEL_LEN=2048 RANDOPT_MAX_NUM_SEQS=128 RANDOPT_MAX_BATCHED_TOKENS=2048
export RANDOPT_MAX_NUM_BATCHED_TOKENS=2048
py=/home/lkx/miniconda3/envs/opcd/bin/python
log="logs/gsm8k_pool_seed20261024_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" scripts/run_resumable.py --max-retries 3 --sleep-seconds 60 -- \
  "$py" scripts/build_score_table.py --dataset gsm8k \
  --model /home/lkx/OPCD/models/Qwen2.5-3B-Instruct \
  --gsm8k-train data/gsm8k/train.parquet --gsm8k-test data/gsm8k/test.parquet \
  --n-candidates 1000 --sigmas 0.0005,0.001,0.002 --pool 1000 --public 50 --test 1319 \
  --seed 20261024 --item-seed 20261023 --max-tokens 1024 --num-engines 1 --tp 1 \
  --precision bfloat16 --gpu-memory-utilization .65 --base-weights-device cpu \
  --checkpoint-every 10 --out "$out" > "$log" 2>&1 < /dev/null &
echo "STARTED wrapper_pid=$! log=$log output=$out"
