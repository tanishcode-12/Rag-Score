"""
Faithfulness - does the generated answer stay grounded in the
retrieved context, or does it introduce claims the context doesn't
support (i.e. hallucinate)? This is the single most requested RAG
metric, and the one number most teams actually act on.

Deliberately scores the answer against the context alone, NOT against
ground_truth_answer - a faithful answer can still be factually wrong
if the context itself was wrong, and that's a retrieval problem, not
a generation problem. Keeping the two separate is what makes it
possible to tell "my retriever found the wrong docs" apart from
"my generator ignored the right docs".
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.judges.base import JudgeVerdict, LLMJudge
from rag_score.metrics.base import Metric

_SYSTEM_PROMPT = """You are a strict, careful evaluator of RAG (Retrieval-Augmented \
Generation) system outputs. Your job is to judge FAITHFULNESS: whether every \
claim in the given answer is actually supported by the given context.

Score from 0.0 to 1.0:
- 1.0: every claim in the answer is directly supported by the context
- 0.5: the answer is partially supported but contains some claims not in the context
- 0.0: the answer contradicts the context or is almost entirely unsupported

Respond with ONLY a JSON object, no other text:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explaining the score>"}"""

_USER_PROMPT_TEMPLATE = """Context:
{context}

Answer to evaluate:
{answer}

Judge whether the answer is fully supported by the context above."""


class Faithfulness(Metric):
    name = "faithfulness"
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
        if not result.generated_answer or not result.retrieved_context:
            return JudgeVerdict(score=0.0, reasoning="No answer or context to evaluate.")

        context_text = "\n\n".join(
            f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(result.retrieved_context)
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            context=context_text, answer=result.generated_answer
        )
        return await self.judge.judge(_SYSTEM_PROMPT, user_prompt)
