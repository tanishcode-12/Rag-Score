"""
Answer Relevance - does the generated answer actually address the
question asked, regardless of whether it's grounded in context?

Complements Faithfulness: an answer can be perfectly faithful to the
context (every claim supported) while still dodging the actual
question (e.g. answering a related-but-different question because
retrieval pulled the wrong chunk). Faithfulness alone would score
that highly; this metric is what catches it.
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.judges.base import JudgeVerdict, LLMJudge
from rag_score.metrics.base import Metric

_SYSTEM_PROMPT = """You are a strict, careful evaluator of RAG (Retrieval-Augmented \
Generation) system outputs. Your job is to judge ANSWER RELEVANCE: whether the \
given answer actually addresses the given question, regardless of whether the \
answer is factually correct or well-supported by any context.

Score from 0.0 to 1.0:
- 1.0: the answer directly and completely addresses the question
- 0.5: the answer is on-topic but partial, vague, or only tangentially responsive
- 0.0: the answer does not address the question at all (e.g. answers a \
different question, or refuses without addressing it)

Respond with ONLY a JSON object, no other text:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explaining the score>"}"""

_USER_PROMPT_TEMPLATE = """Question:
{question}

Answer to evaluate:
{answer}

Judge whether the answer addresses the question above."""


class AnswerRelevance(Metric):
    name = "answer_relevance"
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
        if not result.generated_answer:
            return JudgeVerdict(score=0.0, reasoning="No answer to evaluate.")

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            question=test_case.question, answer=result.generated_answer
        )
        return await self.judge.judge(_SYSTEM_PROMPT, user_prompt)
