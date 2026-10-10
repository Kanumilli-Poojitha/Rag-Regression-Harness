"""Deterministic metrics, aggregation and the regression report."""
import csv
import os
from typing import Dict, List

import numpy as np

REPORT_COLUMNS = ["metric", "baseline", "candidate", "delta", "flag"]
LATENCY_TOLERANCE = 0.10  # candidate may be up to +10% slower before REGRESS


def _norm(t: str) -> str:
    return t.strip().lower()


def recall_at_k(retrieved_titles: List[str], gt_titles: List[str]) -> float:
    gt = {_norm(t) for t in gt_titles}
    if not gt:
        return 1.0
    got = {_norm(t) for t in retrieved_titles}
    return len(gt & got) / len(gt)


def p95(values: List[float]) -> float:
    return float(np.percentile(values, 95))


def aggregate(results: List[Dict]) -> Dict[str, float]:
    m = [r["metrics"] for r in results]
    return {
        "Recall": float(np.mean([x["recall"] for x in m])),
        "Correctness": float(np.mean([x["correctness"] for x in m])),
        "Groundedness": float(np.mean([x["groundedness"] for x in m])),
        "P95_Latency": p95([x["latency_ms"] for x in m]),
    }


def flag_for(metric: str, baseline: float, candidate: float) -> str:
    if metric == "P95_Latency":
        if candidate > baseline * (1 + LATENCY_TOLERANCE):
            return "REGRESS"
        return "IMPROVE" if candidate < baseline else "NEUTRAL"
    if candidate < baseline:
        return "REGRESS"
    return "IMPROVE" if candidate > baseline else "NEUTRAL"


def generate_regression_report(results_a: list, results_b: list,
                               out_path: str = "results/regression_report.csv") -> List[Dict]:
    """Aggregate, compute delta (B - A), apply flags, write CSV. Flags are computed from the
    same rounded values that are written, so the file is self-consistent."""
    agg_a, agg_b = aggregate(results_a), aggregate(results_b)
    rows = []
    for metric in ["Recall", "Correctness", "Groundedness", "P95_Latency"]:
        nd = 2 if metric == "P95_Latency" else 4
        a, b = round(agg_a[metric], nd), round(agg_b[metric], nd)
        rows.append({"metric": metric, "baseline": a, "candidate": b,
                     "delta": round(b - a, nd), "flag": flag_for(metric, a, b)})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def format_table(rows: List[Dict]) -> str:
    lines = [f"{'metric':<14}{'A':>12}{'B':>12}{'delta':>12}  flag"]
    for r in rows:
        lines.append(f"{r['metric']:<14}{r['baseline']:>12}{r['candidate']:>12}{r['delta']:>12}  {r['flag']}")
    return "\n".join(lines)
