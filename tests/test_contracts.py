"""Contract tests for core requirements 1-10. Run after `python -m src.run_eval`.
Result-dependent tests skip (not pass) when results/ has not been generated yet."""
import csv
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
needs_results = pytest.mark.skipif(not (ROOT / "results" / "regression_report.csv").exists(),
                                   reason="run the harness first")


def jload(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_req1_eval_set():
    p = ROOT / "dataset/eval_set.json"
    if not p.exists():
        pytest.skip("run python -m src.build_dataset first")
    data = jload("dataset/eval_set.json")
    assert isinstance(data, list) and 30 <= len(data) <= 50
    for r in data:
        assert set(r) >= {"id", "question", "ground_truth_answer", "ground_truth_context_titles"}
    assert len({r["id"] for r in data}) == len(data)


def test_req2_pins():
    pins = jload("config/eval_pins.json")
    assert isinstance(pins["dataset_size"], (int, float))
    for k in ("llm_judge_model", "pipeline_a_embedder", "pipeline_b_embedder"):
        assert isinstance(pins[k], str) and pins[k]


def test_req3_judge_prompts():
    p = jload("prompts/judge_prompts.json")
    assert p["correctness_prompt"] and p["groundedness_prompt"]
    g = p["groundedness_prompt"].lower()
    assert "context" in g and "any claim" in g and "real world" in g


def _details(label):
    ev = jload("dataset/eval_set.json")
    d = jload(f"results/eval_{label}_details.json")
    assert len(d) == len(ev) and [r["id"] for r in d] == [r["id"] for r in ev]
    for r in d:
        assert isinstance(r["generated_answer"], str)
        assert isinstance(r["retrieved_context_titles"], list)
        m = r["metrics"]
        assert set(m) == {"recall", "correctness", "groundedness", "latency_ms"}
        assert isinstance(m["recall"], float) and 0.0 <= m["recall"] <= 1.0
        assert m["correctness"] in (0, 1) and isinstance(m["correctness"], int)
        assert m["groundedness"] in (0, 1) and isinstance(m["groundedness"], int)
        assert isinstance(m["latency_ms"], float)


@needs_results
def test_req4_details_a():
    _details("A")


@needs_results
def test_req5_details_b():
    _details("B")


def _rows():
    with open(ROOT / "results/regression_report.csv", newline="") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == ["metric", "baseline", "candidate", "delta", "flag"]
        return list(rd)


@needs_results
def test_req6_report_schema():
    rows = _rows()
    assert [r["metric"] for r in rows] == ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    assert all(r["flag"] in ("REGRESS", "IMPROVE", "NEUTRAL") for r in rows)


@needs_results
def test_req7_flag_logic():
    for r in _rows():
        b, c = float(r["baseline"]), float(r["candidate"])
        assert abs(float(r["delta"]) - (c - b)) < 1e-6
        if r["metric"] == "P95_Latency":
            assert (r["flag"] == "REGRESS") == (c > b * 1.10)
        else:
            assert (r["flag"] == "REGRESS") == (c < b)


@needs_results
def test_req8_demonstrated_regression():
    by = {r["metric"]: r for r in _rows()}
    d = {k: float(v["delta"]) for k, v in by.items()}
    assert d["Correctness"] >= 0
    assert d["Groundedness"] < 0 or d["P95_Latency"] > 0
    assert any(r["flag"] == "REGRESS" for r in by.values())


def test_req9_compose():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    vdb = [s for n, s in c["services"].items()
           if any(k in (n + str(s.get("image", ""))).lower() for k in ("qdrant", "weaviate", "chroma", "pgvector"))]
    assert vdb and all("healthcheck" in s for s in vdb)


def test_req10_env_example():
    t = (ROOT / ".env.example").read_text()
    assert "OPENAI_API_KEY" in t and "QDRANT" in t
    assert "sk-" not in t
