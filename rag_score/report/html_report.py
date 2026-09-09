"""
Standalone HTML report generation.

jinja2 is an optional extra (pip install rag-score[report]) so the
core install stays dependency-light; this module only imports it at
call time. Output is a single self-contained .html file with inline
CSS - no server, no external assets, easy to attach to a PR or open
straight from a CI artifact.
"""

from __future__ import annotations

from pathlib import Path

from rag_score.core.runner import RunReport
from rag_score.core.types import DimRun, TestCase

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _score_class(value: float) -> str:
    """Simple traffic-light coloring: >=0.8 good, >=0.5 mid, else bad.
    Kept as a template-callable rather than baked into the data so the
    threshold logic lives in one place."""
    if value >= 0.8:
        return "score-good"
    if value >= 0.5:
        return "score-mid"
    return "score-bad"


def _build_rows(
    report: RunReport, test_cases: list[TestCase]
) -> list[dict]:
    tc_by_id = {tc.test_case_id: tc for tc in test_cases}
    scores_by_eval: dict[str, dict[str, float]] = {}
    for s in report.scores:
        scores_by_eval.setdefault(s.evaluation_id, {})[s.metric_name] = s.score_value

    rows = []
    for result in report.results:
        tc = tc_by_id.get(result.test_case_id)
        total_latency = None
        if result.retrieval_latency_ms is not None and result.generation_latency_ms is not None:
            total_latency = f"{result.retrieval_latency_ms + result.generation_latency_ms:.0f}ms"

        rows.append(
            {
                "question": tc.question if tc else "(unknown question)",
                "generated_answer": result.generated_answer,
                "error": result.error,
                "scores": scores_by_eval.get(result.evaluation_id, {}),
                "total_latency_ms": total_latency or "—",
            }
        )
    return rows


def _summarize(report: RunReport) -> dict[str, float]:
    import statistics

    by_metric: dict[str, list[float]] = {}
    for s in report.scores:
        by_metric.setdefault(s.metric_name, []).append(s.score_value)
    return {name: statistics.mean(values) for name, values in by_metric.items()}


def generate_html_report(
    output_path: str | Path,
    run: DimRun,
    test_cases: list[TestCase],
    report: RunReport,
) -> Path:
    """Render report.html from a completed RunReport. Returns the
    written path."""
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for HTML reports. "
            "Install it with: pip install rag-score[report]"
        ) from e

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    env.globals["score_class"] = _score_class
    template = env.get_template("report.html.j2")

    num_errors = sum(1 for r in report.results if r.error is not None)

    html = template.render(
        run=run,
        num_results=len(report.results),
        num_errors=num_errors,
        summary=_summarize(report),
        rows=_build_rows(report, test_cases),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
