# rag-score

[![Tests](https://github.com/tanishcode-12/Rag-Score/actions/workflows/test.yml/badge.svg)](https://github.com/tanishcode-12/Rag-Score/actions/workflows/test.yml)

Know if your RAG pipeline actually works — in 3 lines of code, with zero framework lock-in.

`rag-score` is a lightweight, framework-agnostic Python library for evaluating Retrieval-Augmented Generation pipelines. Bring your own retriever (LangChain, LlamaIndex, raw FAISS, an HTTP call — anything), get retrieval and generation quality metrics back.

**Core promise: zero lock-in, zero forced API cost.**
- Works with any pipeline via a two-method adapter interface
- Retrieval metrics (Precision@k, Recall@k, MRR, nDCG) run **completely offline** — no API keys needed
- Async-first execution engine so a few hundred test cases don't take hours
- Exports to SQLite (star schema, BI-ready) or Pandas DataFrames for notebooks
- CI/CD ready: writes `$GITHUB_STEP_SUMMARY` automatically in GitHub Actions

## Install

```bash
pip install rag-score
```

## 60-second quickstart

**1. Describe your pipeline as two functions:**

```python
# my_pipeline.py
from rag_score.core.types import RetrievedChunk

async def my_retriever(query: str, top_k: int):
    # call your actual retriever here — FAISS, Pinecone, whatever
    return [RetrievedChunk(doc_id="doc_1", text="...")]

async def my_generator(query: str, context: list[RetrievedChunk]):
    # call your actual LLM here
    return "the generated answer"
```

**2. Write a small test set** (`test_set.json`):

```json
[
  {
    "question": "What is the refund policy?",
    "expected_doc_ids": ["doc_1", "doc_2"]
  }
]
```

**3. Write a config** (`eval_config.json`):

```json
{
  "dataset": "test_set.json",
  "retriever": "my_pipeline:my_retriever",
  "generator": "my_pipeline:my_generator",
  "metrics": ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"],
  "html_output": "report.html"
}
```

**4. Run it:**

```bash
rageval run eval_config.json
```

```
Running 1 test cases with 4 metrics...

Evaluated 1 test cases (0 failed)

Metric          Score
------------------------
precision_at_5  1.000
recall_at_5     0.500
mrr             1.000
ndcg_at_5       0.613

HTML report written to report.html
```

Open `report.html` — no server required, it's a single self-contained file.

## Using it as a library (Jupyter/notebooks)

```python
import asyncio
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.adapters.base import CallableRetrieverAdapter, CallableGeneratorAdapter
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.export.dataframe_export import full_report_dataframe

test_cases = load_dataset("test_set.json")
retriever = CallableRetrieverAdapter(my_retriever)
generator = CallableGeneratorAdapter(my_generator)
metrics = [PrecisionAtK(k=5), MRR()]

report = asyncio.run(
    run_evaluation(test_cases, retriever, generator, metrics, RunConfig(run_id="run-1"))
)

df = full_report_dataframe(report, test_cases)
df.describe()
```

## Offline metrics (v0.1, zero API keys required)

| Metric | What it measures |
|---|---|
| `precision_at_k` | Of the top-k retrieved chunks, what fraction are relevant? |
| `recall_at_k` | Of all relevant chunks, what fraction did top-k retrieval surface? |
| `mrr` | How high up the ranking was the first relevant hit? |
| `ndcg_at_k` | Ranking quality, rewarding relevant results appearing earlier |

LLM-judge metrics (Faithfulness, Answer Relevance — require an API key) are on the roadmap for v0.2.

## BI export

```json
{ "..." : "...", "sqlite_output": "eval_history.db" }
```

Every run appends to the same SQLite file using a star schema (`dim_runs`, `dim_test_cases`, `fact_evaluations`, `fact_metric_scores`), so you can point Power BI, Superset, or a plain SQL query at your evaluation history over time.

## Roadmap

- [ ] LLM-judge metrics: Faithfulness, Answer Relevance, Context Precision (OpenAI/Anthropic/local)
- [ ] LangChain and LlamaIndex adapters (thin wrappers around the base adapter interface)
- [ ] Local-first judges (embedding/NLI models) for offline faithfulness scoring
- [ ] Synthetic test-set generation from raw documents (`rag_score.synthesize`)
- [ ] Agentic trajectory evaluation (multi-step tool calls, routing)

## Why not Ragas / TruLens / DeepEval?

Those are excellent, more full-featured tools. `rag-score` exists for the case where you want something smaller: a library you can read end-to-end in an afternoon, with an adapter interface that doesn't assume you're using any particular framework, and a set of metrics that work with zero API keys before you ever reach for an LLM judge.

## License

MIT