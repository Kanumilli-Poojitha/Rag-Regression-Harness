"""Build dataset/eval_set.json (+ corpus.json) from HotpotQA dev (distractor setting) and
pin its hash in config/eval_pins.json.

    python -m src.build_dataset                       # downloads the official dev file
    python -m src.build_dataset --local-file path.json
    python -m src.build_dataset --if-missing          # no-op when eval_set.json exists
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from . import settings

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_and_convert(raw: list, n: int, seed: int):
    """Pick n multi-hop questions deterministically; return (eval_set, corpus)."""
    usable = []
    for r in raw:
        if str(r["answer"]).strip().lower() in ("yes", "no"):
            continue  # yes/no answers are too easy to match by chance
        titles = []
        for t, _ in r["supporting_facts"]:
            if t not in titles:
                titles.append(t)
        if len(titles) == 2:  # genuine two-document multi-hop
            usable.append((r, titles))
    usable.sort(key=lambda x: x[0]["_id"])
    picked = random.Random(seed).sample(usable, n)
    picked.sort(key=lambda x: x[0]["_id"])

    eval_set, corpus, seen = [], [], set()
    for r, titles in picked:
        eval_set.append({
            "id": r["_id"],
            "question": r["question"],
            "ground_truth_answer": r["answer"],
            "ground_truth_context_titles": titles,
        })
        for title, sents in r["context"]:
            if title not in seen:
                seen.add(title)
                corpus.append({"title": title, "text": " ".join(s.strip() for s in sents).strip()})
    return eval_set, corpus


def update_pins(eval_path, corpus_path, pins_path=settings.PINS_PATH, size=None):
    pins = json.loads(Path(pins_path).read_text(encoding="utf-8"))
    pins["dataset_size"] = size
    pins["dataset_sha256"] = sha256_file(eval_path)
    pins["corpus_sha256"] = sha256_file(corpus_path)
    _dump(pins, pins_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-file")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--if-missing", action="store_true")
    args = ap.parse_args()

    if args.if_missing and settings.EVAL_SET_PATH.exists() and settings.CORPUS_PATH.exists():
        print("eval_set.json already present; skipping build")
        return
    if not 30 <= args.n <= 50:
        raise SystemExit("--n must be between 30 and 50")

    if args.local_file:
        raw = json.loads(Path(args.local_file).read_text(encoding="utf-8"))
    else:
        import requests
        cache = settings.DATASET_DIR / "raw" / "hotpot_dev_distractor_v1.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            print(f"downloading {HOTPOT_URL} ...")
            resp = requests.get(HOTPOT_URL, timeout=300)
            resp.raise_for_status()
            cache.write_bytes(resp.content)
        raw = json.loads(cache.read_text(encoding="utf-8"))

    eval_set, corpus = select_and_convert(raw, args.n, args.seed)
    settings.DATASET_DIR.mkdir(exist_ok=True)
    _dump(eval_set, settings.EVAL_SET_PATH)
    _dump(corpus, settings.CORPUS_PATH)
    update_pins(settings.EVAL_SET_PATH, settings.CORPUS_PATH, size=len(eval_set))
    print(f"wrote {len(eval_set)} questions, {len(corpus)} corpus paragraphs; pins updated")


if __name__ == "__main__":
    main()
