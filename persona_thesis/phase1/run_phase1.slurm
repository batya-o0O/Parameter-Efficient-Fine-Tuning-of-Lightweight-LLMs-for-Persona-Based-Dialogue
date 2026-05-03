#!/bin/bash
#SBATCH -J phase1_rolebench_full
#SBATCH -p H100
#SBATCH --qos=nu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

mkdir -p logs
cd ~/persona_thesis/phase1

source .venv/bin/activate

echo "Node: $(hostname)"
nvidia-smi || true
python -V

# Pick ONE precision mode:
# For H100, bf16 is best:
python train_rolebench_phase1.py \
  --output_dir ./outputs/phi3_rolebench_phase1_full \
  --epochs 1 \
  --max_length 1024 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --save_steps 2000 \
  --eval_steps 2000 \
  --logging_steps 50 \
  --save_total_limit 3 \
  --dataloader_workers 4 \
  --group_by_length \
  --packing \
  --report_to tensorboard \
  --bf16
