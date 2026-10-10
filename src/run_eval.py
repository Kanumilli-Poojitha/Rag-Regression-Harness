"""Run pipelines A and B on the pinned eval set, judge them, write all artifacts.

    python -m src.run_eval            (or: python run_eval.py)
"""
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

from . import settings
from .build_dataset import sha256_file
from .judge import Judge
from .metrics import format_table, generate_regression_report, p95, recall_at_k


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_pins(pins, eval_set, cfg_a, cfg_b):
    problems = []
    if not pins.get("dataset_sha256"):
        problems.append("pins.dataset_sha256 empty - run `python -m src.build_dataset` first")
    elif sha256_file(settings.EVAL_SET_PATH) != pins["dataset_sha256"]:
        problems.append("eval_set.json hash differs from pins.dataset_sha256 (dataset moved)")
    if pins.get("corpus_sha256") and sha256_file(settings.CORPUS_PATH) != pins["corpus_sha256"]:
        problems.append("corpus.json hash differs from pins.corpus_sha256")
    if len(eval_set) != pins["dataset_size"]:
        problems.append("dataset_size pin does not match eval_set length")
    for label, cfg, key in (("A", cfg_a, "pipeline_a_embedder"), ("B", cfg_b, "pipeline_b_embedder")):
        if cfg["embedder"] != pins[key]:
            problems.append(f"pipeline {label} embedder {cfg['embedder']} != pin {pins[key]}")
        if cfg["generator_model"] != pins["generator_model"]:
            problems.append(f"pipeline {label} generator model != pinned generator_model")
        if cfg["chunker_version"] != pins["chunker_version"]:
            problems.append(f"pipeline {label} chunker_version != pin")
    if cfg_b.get("reranker") != pins.get("pipeline_b_reranker"):
        problems.append("pipeline B reranker != pin")
    if problems:
        raise SystemExit("PIN CHECK FAILED:\n  - " + "\n  - ".join(problems))


def main():
    from .pipeline import RAGPipeline  # heavy imports deferred

    skip_ingest = "--skip-ingest" in sys.argv
    pins = _load(settings.PINS_PATH)
    eval_set = _load(settings.EVAL_SET_PATH)
    corpus = _load(settings.CORPUS_PATH)
    path_a, path_b = str(settings.CONFIG_DIR / "pipeline_a.json"), str(settings.CONFIG_DIR / "pipeline_b.json")
    verify_pins(pins, eval_set, _load(path_a), _load(path_b))

    pipe_a, pipe_b = RAGPipeline(path_a), RAGPipeline(path_b)
    if not skip_ingest:
        for p in (pipe_a, pipe_b):
            print(f"[ingest] {p.name}: {p.ingest(corpus)} chunks")
    for p in (pipe_a, pipe_b):  # warm-up so model load/first-call cost is not in latency
        p.retrieve(eval_set[0]["question"])

    judge = Judge(pins["llm_judge_model"])
    details = {"A": [], "B": []}
    traces = []

    for idx, item in enumerate(eval_set):
        order = [("A", pipe_a), ("B", pipe_b)] if idx % 2 == 0 else [("B", pipe_b), ("A", pipe_a)]
        for label, pipe in order:  # alternate order so API drift does not favour one side
            out = pipe.retrieve_and_generate(item["question"])
            corr, corr_raw = judge.correctness(item["question"], item["ground_truth_answer"], out["answer"])
            grd, grd_raw = judge.groundedness(out["retrieved_contexts"], out["answer"])
            rec = recall_at_k(out["context_titles"], item["ground_truth_context_titles"])
            details[label].append({
                "id": item["id"],
                "generated_answer": out["answer"],
                "retrieved_context_titles": out["context_titles"],
                "metrics": {
                    "recall": float(rec),
                    "correctness": int(corr),
                    "groundedness": int(grd),
                    "latency_ms": float(out["latency_ms"]),
                },
            })
            traces.append({"pipeline": label, "id": item["id"], "question": item["question"],
                           "ground_truth_answer": item["ground_truth_answer"],
                           "retrieved_contexts": out["retrieved_contexts"],
                           "judge_raw": {"correctness": corr_raw, "groundedness": grd_raw}})
        print(f"[{idx + 1}/{len(eval_set)}] done", flush=True)

    # keep details in dataset order regardless of execution order
    order_of = {q["id"]: i for i, q in enumerate(eval_set)}
    for k in details:
        details[k].sort(key=lambda r: order_of[r["id"]])

    settings.RESULTS_DIR.mkdir(exist_ok=True)
    for label in ("A", "B"):
        with open(settings.RESULTS_DIR / f"eval_{label}_details.json", "w", encoding="utf-8") as f:
            json.dump(details[label], f, indent=2, ensure_ascii=False)
    with open(settings.RESULTS_DIR / "judge_trace.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    rows = generate_regression_report(details["A"], details["B"], str(settings.RESULTS_DIR / "regression_report.csv"))

    lat = {k: [r["metrics"]["latency_ms"] for r in v] for k, v in details.items()}
    meta = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "pins": pins,
        "pipeline_configs": {"A": _load(path_a), "B": _load(path_b)},
        "latency_ms": {k: {"p50": float(np.percentile(v, 50)), "p95": p95(v)} for k, v in lat.items()},
        "hardware_note": {"platform": platform.platform(), "python": platform.python_version(),
                          "cpu_count": os.cpu_count(), "gpu_used": False,
                          "note": "Latency includes the hosted-LLM network round trip; compare only runs on the same hardware."},
    }
    with open(settings.RESULTS_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + format_table(rows))
    if not any(r["flag"] == "REGRESS" for r in rows):
        print("\nWARNING: no REGRESS flag. Requirement 8 needs B to regress on groundedness or latency "
              "while correctness holds; see README 'Tuning the regression'.")


if __name__ == "__main__":
    main()
