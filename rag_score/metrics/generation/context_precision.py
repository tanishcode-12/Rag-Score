"""
Context Precision - of the chunks that were retrieved, how many are
actually relevant to the question? Unlike precision_at_k (which needs
labeled expected_doc_ids and pure set math), this uses an LLM judge to
assess relevance directly from content - useful when you don't have
(or trust) hand-labeled relevance judgments, or want a second signal
that doesn't depend on doc IDs matching exactly.

v1 makes one holistic judge call per test case (not one call per
chunk) to keep cost/latency down - the judge sees all retrieved chunks
together and estimates what fraction are relevant. Per-chunk grading
is a natural Phase 3 refinement if the holistic estimate proves too
coarse in practice.
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.judges.base import JudgeVerdict, LLMJudge
from rag_score.metrics.base import Metric

_SYSTEM_PROMPT = """You are a strict, careful evaluator of RAG (Retrieval-Augmented \
Generation) retrieval quality. Your job is to judge CONTEXT PRECISION: of the \
numbered context chunks retrieved for the question, what fraction are actually \
relevant and useful for answering it?

Score from 0.0 to 1.0, representing the fraction of chunks that are relevant:
- 1.0: every chunk is relevant and useful for answering the question
- 0.5: about half the chunks are relevant; the rest are off-topic or unhelpful
- 0.0: none of the chunks are relevant to the question

Respond with ONLY a JSON object, no other text:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explaining the score>"}"""

_USER_PROMPT_TEMPLATE = """Question:
{question}

Retrieved context chunks:
{context}

Judge what fraction of these chunks are relevant to answering the question."""


class ContextPrecision(Metric):
    name = "context_precision"
    requires_api_key = True

    def __init__(self, judge: LLMJudge) -> None:
        self.judge = judge

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        verdict = await self._verdict(test_case, result)
        return verdict.score

    async def score_with_reasoning(
        self, test_case: TestCase, result: EvalResult
    ) -> tuple[float, str | None]:
        verdict = await self._verdict(test_case, result)
        return verdict.score, verdict.reasoning

    async def _verdict(self, test_case: TestCase, result: EvalResult) -> JudgeVerdict:
        if not result.retrieved_context:
            return JudgeVerdict(score=0.0, reasoning="No retrieved context to evaluate.")

        context_text = "\n\n".join(
            f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(result.retrieved_context)
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            question=test_case.question, context=context_text
        )
        return await self.judge.judge(_SYSTEM_PROMPT, user_prompt)
