"""
Recall@k — of all the documents that are actually relevant, what
fraction did the top-k retrieval surface? Complements precision@k:
precision asks "how clean is what I got back", recall asks "how much
of the good stuff did I miss".
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.metrics.base import Metric


class RecallAtK(Metric):
    name = "recall_at_k"
    requires_api_key = False

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.name = f"recall_at_{k}"

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        expected = set(test_case.expected_doc_ids)
        if not expected:
            return 0.0

        top_k = result.retrieved_context[: self.k]
        retrieved_ids = {chunk.doc_id for chunk in top_k if chunk.doc_id is not None}

        relevant_found = len(expected & retrieved_ids)
        return relevant_found / len(expected)
