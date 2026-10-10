"""Offline unit tests (no API, no Qdrant)."""
import csv
import json

import pytest

from src.build_dataset import select_and_convert
from src.chunking import chunk_words
from src.judge import JudgeError, parse_binary
from src.metrics import flag_for, generate_regression_report, p95, recall_at_k


def test_chunker():
    assert chunk_words("a b c d e", 2) == ["a b", "c d", "e"]
    assert chunk_words("a b c d", 2, 1) == ["a b", "b c", "c d"]
    assert chunk_words("", 3) == []


def test_recall():
    assert recall_at_k(["A", "x"], ["A", "B"]) == 0.5
    assert recall_at_k(["a", "b"], ["A", "B"]) == 1.0


def test_parse_binary():
    assert parse_binary(" 1\n") == 1
    for bad in ("1.", "yes", ""):
        with pytest.raises(JudgeError):
            parse_binary(bad)


@pytest.mark.parametrize("metric,b,c,expected", [
    ("Groundedness", 0.9, 0.8, "REGRESS"), ("Correctness", 0.5, 0.6, "IMPROVE"),
    ("Recall", 0.5, 0.5, "NEUTRAL"),
    ("P95_Latency", 1000, 1100, "NEUTRAL"), ("P95_Latency", 1000, 1100.01, "REGRESS"),
    ("P95_Latency", 1000, 900, "IMPROVE"),
])
def test_flags(metric, b, c, expected):
    assert flag_for(metric, b, c) == expected


def _mk(rec, cor, grd, lat):
    return [{"id": str(i), "metrics": {"recall": rec, "correctness": cor[i], "groundedness": grd[i], "latency_ms": lat[i]}}
            for i in range(len(cor))]


def test_report(tmp_path):
    a = _mk(0.5, [1, 0, 1, 0], [1, 1, 1, 1], [100, 110, 120, 130])
    b = _mk(1.0, [1, 1, 1, 0], [1, 0, 1, 0], [300, 310, 320, 330])
    out = tmp_path / "r.csv"
    rows = generate_regression_report(a, b, str(out))
    got = list(csv.DictReader(open(out)))
    assert [r["metric"] for r in got] == ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    by = {r["metric"]: r for r in got}
    assert by["Correctness"]["flag"] == "IMPROVE"
    assert by["Groundedness"]["flag"] == "REGRESS"
    assert by["P95_Latency"]["flag"] == "REGRESS"
    assert float(by["Groundedness"]["delta"]) == -0.5
    assert float(by["P95_Latency"]["baseline"]) == round(p95([100, 110, 120, 130]), 2)
    assert len(rows) == 4


def test_dataset_selection():
    raw = []
    for i in range(60):
        raw.append({
            "_id": f"id{i:03d}", "question": f"q{i}?", "answer": "No" if i % 10 == 0 else f"ans{i}",
            "supporting_facts": [[f"T{i}a", 0], [f"T{i}b", 1], [f"T{i}a", 2]],
            "context": [[f"T{i}a", ["s1.", "s2."]], [f"T{i}b", ["s3."]], [f"D{i}", ["x"]]],
        })
    e1, c1 = select_and_convert(raw, 30, 42)
    e2, c2 = select_and_convert(raw, 30, 42)
    assert e1 == e2 and c1 == c2 and len(e1) == 30
    assert all(r["ground_truth_answer"].lower() != "no" for r in e1)
    assert set(e1[0]) == {"id", "question", "ground_truth_answer", "ground_truth_context_titles"}
    assert len(e1[0]["ground_truth_context_titles"]) == 2
