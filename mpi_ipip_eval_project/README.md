# MPI/IPIP-style OCEAN Evaluation for Persona LLMs

Run actual models:
```bash
bash scripts/run_base_and_lora.sh
bash scripts/run_controllability.sh
python src/compare_mpi_results.py --results_dir results --target_profile configs/target_profile_example.json --output_dir results/summary
```

Smoke test:
```bash
bash scripts/run_demo_outputs.sh
```
