# RAG Evaluation Regression Harness

Compares RAG pipeline **A** (baseline: fixed-size chunks, top-2) with **B** (candidate: top-20 -> cross-encoder rerank -> top-5,
"explain" answer style) on a pinned 40-question HotpotQA slice. Recall@k, correctness, groundedness and p95 latency are scored
independently, and `results/regression_report.csv` flags regressions.

## Reproduce
```bash
cp .env.example .env            # set OPENAI_API_KEY
docker compose up --build --abort-on-container-exit
# or locally:
docker compose up -d qdrant && pip install -r requirements.txt
python -m src.build_dataset     # downloads HotpotQA dev (distractor), writes dataset/*.json, pins hash
python -m src.run_eval          # writes everything in results/
pytest                          # contract tests for requirements 1-10
```
If the CMU download is blocked, pass `--local-file hotpot_dev_distractor_v1.json` to `build_dataset`.

## Layout
| Path | Purpose |
|---|---|
| `dataset/eval_set.json`, `dataset/corpus.json` | frozen questions + shared paragraph corpus (seed 42) |
| `config/eval_pins.json` | dataset sha256, judge/generator models, embedders, reranker, chunker version; `run_eval` aborts if anything drifts |
| `config/pipeline_{a,b}.json` | the only differences between A and B |
| `prompts/judge_prompts.json` | correctness + groundedness rubrics (few-shot, binary output) |
| `results/eval_{A,B}_details.json` | per-question metrics (exact schema) |
| `results/judge_trace.jsonl` | retrieved contexts + raw judge outputs, so every score can be audited |
| `results/run_metadata.json` | p50/p95, hardware note, full pins |
| `results/regression_report.csv` | `metric,baseline,candidate,delta,flag` |

## Metrics
- **Recall**: fraction of ground-truth titles present among retrieved titles (deterministic).
- **Correctness / Groundedness**: independent LLM-judge calls, temperature 0, output must be exactly `0` or `1` (anything else is retried, then fails loudly).
- **P95_Latency**: end-to-end ms (query in -> answer out, retrieval + rerank + generation), judge excluded; models are warmed up first and A/B order alternates per question.
- **Flags**: quality metrics REGRESS if candidate < baseline; latency REGRESS if candidate > baseline x 1.10.

## Tuning the regression (requirement 8)
B differs from A in three ways: more chunks, a reranker, and a more "explanatory" generation prompt (`answer_style: explain`).
Reranking adds latency almost deterministically; the groundedness drop comes from the explanation style pulling in tangential
chunks or background knowledge. If a run shows no regression, raise `top_k` in `config/pipeline_b.json`, then re-run
(and note the change - pins and configs are part of the experiment).
Fixtures: the failure class covered is "accuracy up, groundedness/latency down".

## Notes
- Latency includes the hosted-LLM network round trip; compare only runs on the same machine/network.
- Small sample (40): one question moves a quality metric by 0.025. Treat tiny deltas as noise.
