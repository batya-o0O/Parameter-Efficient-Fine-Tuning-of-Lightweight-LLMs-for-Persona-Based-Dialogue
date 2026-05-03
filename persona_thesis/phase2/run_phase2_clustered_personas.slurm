#!/bin/bash
#SBATCH -J phase2_clustered_personas_k10
#SBATCH -p H100
#SBATCH --qos=nu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH --array=0-9%4
#SBATCH -o phase2/logs/%x_%A_%a.out
#SBATCH -e phase2/logs/%x_%A_%a.err

set -euo pipefail

cd ~/persona_thesis
mkdir -p phase2/logs

# Reuse the existing training environment from phase1.
source phase1/.venv/bin/activate

export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/hf_datasets}"
export CLUSTERS_FILE="data/clusters_k10.json"

echo "Node: $(hostname)"
nvidia-smi || true
python -V
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

CLUSTER_ID="$(python - <<'PY'
import json
import os

path = os.environ["CLUSTERS_FILE"]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(path, "r", encoding="utf-8") as f:
    clusters = json.load(f)["clusters"]
ids = sorted(clusters.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
if idx < 0 or idx >= len(ids):
    raise SystemExit(f"SLURM_ARRAY_TASK_ID={idx} out of range for {len(ids)} clusters")
print(ids[idx], end="")
PY
)"
echo "Array task ${SLURM_ARRAY_TASK_ID} -> cluster ${CLUSTER_ID}"

python src/train/train_cluster_adapters.py \
  --clusters "$CLUSTERS_FILE" \
  --cluster_ids "$CLUSTER_ID" \
  --output_root phase2/outputs/phi3_rolebench_phase2_clusters_k10 \
  --model_id microsoft/Phi-3-mini-4k-instruct \
  --dataset_repo ZenMoore/RoleBench \
  --train_file rolebench-eng/role-generalization/general/train.jsonl \
  --eval_file rolebench-eng/role-generalization/general/test.jsonl \
  --epochs 1 \
  --max_length 1024 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --save_steps 2000 \
  --eval_steps 2000 \
  --logging_steps 50 \
  --save_total_limit 2 \
  --dataloader_workers 4 \
  --group_by_length \
  --packing \
  --report_to tensorboard \
  --use_bf16
