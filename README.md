# Parameter-Efficient Fine-Tuning of Lightweight LLMs for Persona-Based Dialogue

Code and released adapter artifacts for a master's thesis on parameter-efficient fine-tuning of lightweight language models for persona-based dialogue.

## Overview

This repository collects the main code used across the project:

- RoleBench clustering and clustered LoRA training
- full-dataset phase1 LoRA training
- RoleBench evaluation scripts
- CharacterBench evaluation scripts
- PersonaGym generation and evaluation scripts
- Kaggle notebooks used for generation and clustering experiments

It also includes the final released adapter artifacts used in the main experiments.

## Repository Structure

- `persona_thesis/`
  Original training and clustering code.
- `persona_thesis_rtx4500_k5/`
  Packaged server-side workflow for 5-cluster training and evaluation.
- `code/`
  Additional evaluation scripts and Kaggle notebooks.
- `code/CharacterBench/`
  CharacterBench-related scripts and prompt construction utilities used in this project.
- `PersonaGym-master/PersonaGym-master/`
  Local PersonaGym evaluation code, prompts, and rubrics used by the thesis workflow.
- `adapters/`
  Final released LoRA adapters.

## Included Adapters

The repository contains the final adapter payloads only, without intermediate checkpoints:

- `adapters/phi3_rolebench_phase1_full/`
  Full-dataset phase1 LoRA adapter.
- `adapters/cluster_0/`
- `adapters/cluster_1/`
- `adapters/cluster_2/`
- `adapters/cluster_3/`
- `adapters/cluster_4/`
  Final 5-cluster RoleBench LoRA adapters.

These folders include the adapter weights and the metadata needed to load them.

## Main Workflows

### 1. Clustering and training

The original clustering and training code is under:

- `persona_thesis/src/`
- `persona_thesis/phase1/`
- `persona_thesis/phase2/`

The packaged server workflow is under:

- `persona_thesis_rtx4500_k5/`

Key entry points:

- `persona_thesis/phase1/train_rolebench_phase1.py`
- `persona_thesis/src/train/train_cluster_adapters.py`
- `persona_thesis_rtx4500_k5/run_train_k5.sh`

### 2. RoleBench evaluation

Key scripts:

- `persona_thesis_rtx4500_k5/src/eval/evaluate_rolebench_paper_routed.py`
- `persona_thesis_rtx4500_k5/run_eval_paper_base.sh`
- `persona_thesis_rtx4500_k5/run_eval_paper_phase1_adapter.sh`
- `persona_thesis_rtx4500_k5/run_eval_paper_k5.sh`

### 3. CharacterBench evaluation

Key scripts:

- `persona_thesis_rtx4500_k5/src/eval/generate_characterbench_responses.py`
- `persona_thesis_rtx4500_k5/src/eval/evaluate_characterbench_judge.py`
- `persona_thesis_rtx4500_k5/run_characterbench_compare.sh`
- `persona_thesis_rtx4500_k5/run_characterbench_phase1_adapter.sh`
- `persona_thesis_rtx4500_k5/run_characterbench_routed.sh`

### 4. PersonaGym evaluation

Key scripts:

- `code/personagym_experiment1_kaggle.ipynb`
- `code/personagym_partial_eval.py`
- `code/personagym_full_eval.py`

The local PersonaGym evaluator in this repo is configured to use `gpt-5-nano` by default unless overridden with environment variables.

## Notes On Excluded Files

This repository intentionally excludes:

- raw result folders
- downloaded evaluation outputs
- server logs
- training checkpoints
- local API secrets
- large benchmark dumps that were not necessary to release the code

Only the final adapter artifacts and the runnable code/scripts are included.

## API Keys

The real PersonaGym API key file is not committed.

Use:

- `PersonaGym-master/PersonaGym-master/code/api_keys.example.py`

as the template for creating your own local:

- `PersonaGym-master/PersonaGym-master/code/api_keys.py`

## Environment

Most training and evaluation code in this repository assumes:

- Python 3.10+
- PyTorch + Transformers + PEFT
- optional 4-bit quantized loading with BitsAndBytes
- GPU execution for generation and adapter evaluation

See the package-specific files for dependencies and launch scripts:

- `persona_thesis_rtx4500_k5/requirements.txt`
- `PersonaGym-master/PersonaGym-master/requirements.txt`

## License And Attribution

This repository contains original thesis code plus local copies of third-party benchmark/evaluation utilities used during the experiments. Please check the corresponding subfolder licenses and READMEs where applicable.
