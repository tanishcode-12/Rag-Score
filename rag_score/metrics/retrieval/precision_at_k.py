"""
Precision@k — of the top-k retrieved chunks, what fraction are actually
relevant (i.e. their doc_id appears in the TestCase's expected_doc_ids)?

Pure set-membership math, no dependencies, no network calls — this is
one of the "offline-first" metrics that works with zero API keys.
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.metrics.base import Metric


class PrecisionAtK(Metric):
    name = "precision_at_k"
    requires_api_key = False

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        # Keep the metric name specific per-k so a report can show
        # precision_at_5 and precision_at_10 side by side.
        self.name = f"precision_at_{k}"

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        expected = set(test_case.expected_doc_ids)
        if not expected:
            # No ground truth to compare against -> undefined, not zero.
            # Callers/report should treat None-like 0.0 as "skipped".
            return 0.0

        top_k = result.retrieved_context[: self.k]
        if not top_k:
            return 0.0

        retrieved_ids = [chunk.doc_id for chunk in top_k if chunk.doc_id is not None]
        if not retrieved_ids:
            # Adapter didn't supply doc_ids at all - precision can't be computed.
            return 0.0

        relevant_count = sum(1 for doc_id in retrieved_ids if doc_id in expected)
        return relevant_count / len(top_k)
