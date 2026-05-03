#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated_tables_current"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def fmt_sci(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}e}"


def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_latex_table(path: Path, rows: list[dict], fieldnames: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(fieldnames)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(latex_escape(name) for name in fieldnames) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(name, "")) for name in fieldnames) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def find_latest_trainer_state(cluster_dir: Path) -> Path | None:
    candidates = sorted(
        cluster_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.replace("checkpoint-", "")),
    )
    return candidates[-1] if candidates else None


def build_cluster_training_rows() -> list[dict]:
    rows = []
    for cluster_dir in sorted((ROOT / "downloads").glob("cluster_*")):
        meta = load_json(cluster_dir / "cluster_meta.json")
        trainer_state_path = find_latest_trainer_state(cluster_dir)
        state = load_json(trainer_state_path) if trainer_state_path else {}
        log_history = [entry for entry in state.get("log_history", []) if "loss" in entry]
        last = log_history[-1] if log_history else {}
        rows.append(
            {
                "cluster": cluster_dir.name,
                "roles": len(meta.get("roles", [])),
                "train_rows": meta.get("train_rows", ""),
                "eval_rows": meta.get("eval_rows", ""),
                "final_checkpoint": trainer_state_path.parent.name if trainer_state_path else "",
                "global_step": state.get("global_step", ""),
                "total_flos": fmt_sci(state.get("total_flos")),
                "last_logged_step": last.get("step", ""),
                "last_logged_loss": fmt_float(last.get("loss")),
                "last_logged_entropy": fmt_float(last.get("entropy")),
                "last_logged_token_accuracy": fmt_float(last.get("mean_token_accuracy")),
                "last_logged_num_tokens": int(last.get("num_tokens", 0)) if last.get("num_tokens") is not None else "",
            }
        )
    return rows


def build_phase1_training_rows() -> list[dict]:
    state = load_json(ROOT / "phi3_rolebench_phase1_full" / "checkpoint-669" / "trainer_state.json")
    log_text = (ROOT / "persona_thesis" / "phase1" / "logs" / "phase1_rolebench_full_7983.out").read_text(encoding="utf-8")

    def extract(pattern: str):
        match = re.search(pattern, log_text)
        return match.group(1) if match else ""

    return [
        {
            "run": "phase1_global_adapter",
            "train_examples": extract(r"Train examples:\s+(\d+)"),
            "eval_examples": "5000" if re.search(r"num_rows:\s+5000", log_text) else "",
            "effective_batch_size": extract(r"Effective batch size:\s+(\d+)"),
            "steps_per_epoch_estimate": extract(r"Steps/epoch:\s+(\d+)"),
            "global_step_saved": state.get("global_step", ""),
            "total_flos": fmt_sci(state.get("total_flos")),
            "last_logged_loss": fmt_float([entry for entry in state["log_history"] if "loss" in entry][-1]["loss"]),
            "last_logged_entropy": fmt_float([entry for entry in state["log_history"] if "loss" in entry][-1]["entropy"]),
            "last_logged_token_accuracy": fmt_float([entry for entry in state["log_history"] if "loss" in entry][-1]["mean_token_accuracy"]),
            "last_logged_num_tokens": int([entry for entry in state["log_history"] if "loss" in entry][-1]["num_tokens"]),
            "final_train_runtime_s": extract(r"'train_runtime':\s+([0-9.]+)"),
            "final_train_loss": extract(r"'train_loss':\s+([0-9.]+)"),
            "final_num_tokens": extract(r"'num_tokens':\s+([0-9.]+), 'mean_token_accuracy'"),
            "final_mean_token_accuracy": extract(r"'mean_token_accuracy':\s+([0-9.]+)"),
        }
    ]


def build_rolebench_rows() -> list[dict]:
    specs = [
        ("cluster0_smoke", ROOT / "downloads" / "evals" / "rolebench_paper_eval_cluster0.json"),
        ("routed_k5", ROOT / "downloads" / "evals" / "rolebench_paper_eval_k5.json"),
        ("base_partial", ROOT / "downloads" / "evals" / "rolebench_paper_eval_base_partial.json"),
        ("phase1_adapter", ROOT / "downloads" / "evals" / "phase1" / "rolebench_paper_eval_phase1_adapter.json"),
    ]
    rows = []
    for tag, path in specs:
        payload = load_json(path)
        results = payload["results"]
        main_key = next(key for key in results.keys() if key != "rolegpt_baseline")
        main = results[main_key]
        baseline = results.get("rolegpt_baseline", {})
        rows.append(
            {
                "run": tag,
                "eval_mode": payload["metadata"].get("eval_mode", "routed"),
                "benchmark_root": payload["metadata"].get("benchmark_root", ""),
                "general_examples": payload["evaluation_data"].get("general_examples", ""),
                "role_specific_examples": payload["evaluation_data"].get("role_specific_examples", ""),
                "missing_raw_matches": payload["evaluation_data"].get("missing_raw_matches", ""),
                "label": main.get("label", ""),
                "spe": fmt_float(main.get("metrics", {}).get("SPE")),
                "avg": fmt_float(main.get("metrics", {}).get("AVG")),
                "baseline_spe": fmt_float(baseline.get("metrics", {}).get("SPE")),
                "baseline_avg": fmt_float(baseline.get("metrics", {}).get("AVG")),
            }
        )
    return rows


def count_response_rows(path: Path) -> tuple[int, int]:
    files = sorted(path.glob("*.json"))
    total = 0
    for file_path in files:
        total += len(load_json(file_path))
    return len(files), total


def build_characterbench_summary_rows() -> list[dict]:
    runs = [
        ("base_model_labeled", ROOT / "downloads" / "evals" / "characterbench_summary_base_model.json", ROOT / "downloads" / "response_data_phi3_base_correct"),
        ("base_model_legacy", ROOT / "downloads" / "evals" / "characterbench_summary.json", ROOT / "downloads" / "response_data_phi3_base_correct"),
        ("cluster0_adapter", ROOT / "downloads" / "evals" / "characterbench_summary_adapter.json", ROOT / "downloads" / "evals" / "response_data_cluster_0_adapter"),
        ("routed_k5_profiles", ROOT / "downloads" / "evals" / "characterbench_summary_adapter_new.json", None),
        ("phase1_global_adapter", ROOT / "downloads" / "evals" / "phase1" / "characterbench_summary.json", ROOT / "downloads" / "evals" / "phase1" / "response_data_phi3_phase1_full"),
    ]
    rows = []
    for run_name, summary_path, response_dir in runs:
        payload = load_json(summary_path)
        file_count, row_count = ("", "")
        if response_dir and response_dir.exists():
            file_count, row_count = count_response_rows(response_dir)
        rows.append(
            {
                "run": run_name,
                "average": fmt_float(payload.get("average")),
                "response_files": file_count,
                "response_rows": row_count,
            }
        )
    return rows


def build_characterbench_metric_rows() -> list[dict]:
    run_payloads = {
        "base_model": load_json(ROOT / "downloads" / "evals" / "characterbench_summary_base_model.json"),
        "cluster0_adapter": load_json(ROOT / "downloads" / "evals" / "characterbench_summary_adapter.json"),
        "routed_k5_profiles": load_json(ROOT / "downloads" / "evals" / "characterbench_summary_adapter_new.json"),
        "phase1_global_adapter": load_json(ROOT / "downloads" / "evals" / "phase1" / "characterbench_summary.json"),
    }

    def find_metric(payload: dict, metric_suffix: str):
        for key, value in payload.items():
            if key == "average":
                continue
            if key.endswith(metric_suffix):
                return value
        return None

    metrics = [
        "attribute_consistency_bot_test",
        "attribute_consistency_human_test",
        "behavior_consistency_bot_test",
        "behavior_consistency_human_test",
        "boundary_consistency_test",
        "emotion_self_regulation_test",
        "empathetic_responsiveness_test",
        "engagement_test",
        "fact_accuracy_test",
        "human_likeness_test",
        "memory_consistency_test",
        "morality_robustness_test",
        "morality_stability_test",
        "average",
    ]

    rows = []
    for metric in metrics:
        row = {"metric": metric}
        for run_name, payload in run_payloads.items():
            value = payload.get("average") if metric == "average" else find_metric(payload, metric)
            row[run_name] = fmt_float(value)
        rows.append(row)
    return rows


def write_all_tables():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cluster_rows = build_cluster_training_rows()
    cluster_fields = [
        "cluster",
        "roles",
        "train_rows",
        "eval_rows",
        "final_checkpoint",
        "global_step",
        "total_flos",
        "last_logged_step",
        "last_logged_loss",
        "last_logged_entropy",
        "last_logged_token_accuracy",
        "last_logged_num_tokens",
    ]
    write_csv(OUT_DIR / "cluster_training_summary.csv", cluster_rows, cluster_fields)
    write_latex_table(
        OUT_DIR / "cluster_training_summary.tex",
        cluster_rows,
        cluster_fields,
        caption="Clustered RoleBench adapter training summary.",
        label="tab:cluster-training-summary",
    )

    phase1_rows = build_phase1_training_rows()
    phase1_fields = list(phase1_rows[0].keys())
    write_csv(OUT_DIR / "phase1_training_summary.csv", phase1_rows, phase1_fields)
    write_latex_table(
        OUT_DIR / "phase1_training_summary.tex",
        phase1_rows,
        phase1_fields,
        caption="Phase1 global adapter training summary.",
        label="tab:phase1-training-summary",
    )

    rolebench_rows = build_rolebench_rows()
    rolebench_fields = list(rolebench_rows[0].keys())
    write_csv(OUT_DIR / "rolebench_results_summary.csv", rolebench_rows, rolebench_fields)
    write_latex_table(
        OUT_DIR / "rolebench_results_summary.tex",
        rolebench_rows,
        rolebench_fields,
        caption="Saved RoleBench paper-style evaluation summaries.",
        label="tab:rolebench-results-summary",
    )

    characterbench_rows = build_characterbench_summary_rows()
    characterbench_fields = list(characterbench_rows[0].keys())
    write_csv(OUT_DIR / "characterbench_summary.csv", characterbench_rows, characterbench_fields)
    write_latex_table(
        OUT_DIR / "characterbench_summary.tex",
        characterbench_rows,
        characterbench_fields,
        caption="CharacterBench pilot-set summary results.",
        label="tab:characterbench-summary",
    )

    metric_rows = build_characterbench_metric_rows()
    metric_fields = list(metric_rows[0].keys())
    write_csv(OUT_DIR / "characterbench_metric_comparison.csv", metric_rows, metric_fields)
    write_latex_table(
        OUT_DIR / "characterbench_metric_comparison.tex",
        metric_rows,
        metric_fields,
        caption="CharacterBench metric-by-metric comparison across models.",
        label="tab:characterbench-metric-comparison",
    )


if __name__ == "__main__":
    write_all_tables()
    print(f"Wrote tables to: {OUT_DIR}")
