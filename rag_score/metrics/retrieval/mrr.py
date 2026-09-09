"""
MRR (Mean Reciprocal Rank) for a single query is just the Reciprocal
Rank of the first relevant hit: 1/rank if found, else 0. The runner's
report averages these across the dataset to get the actual "mean".
Named RReciprocalRank internally to be accurate at the single-result
level, but exposed with metric name "mrr" since that's what the
report/CLI/users expect to see.
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.metrics.base import Metric


class MRR(Metric):
    name = "mrr"
    requires_api_key = False

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        expected = set(test_case.expected_doc_ids)
        if not expected:
            return 0.0

        for rank, chunk in enumerate(result.retrieved_context, start=1):
            if chunk.doc_id is not None and chunk.doc_id in expected:
                return 1.0 / rank

        return 0.0
