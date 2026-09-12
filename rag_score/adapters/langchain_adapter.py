"""
LangChain adapter.

Wraps a LangChain BaseRetriever and a Runnable (chain, chat model,
LCEL pipeline - anything with .ainvoke) so an existing LangChain RAG
pipeline can be evaluated with zero rewriting: pass your retriever and
chain straight in.

langchain-core is an optional dependency - only imported when these
classes are actually instantiated, so installing rag-score doesn't
pull in LangChain for people who don't use it.
"""

from __future__ import annotations

from typing import Any

from rag_score.adapters.base import GeneratorAdapter, RetrieverAdapter
from rag_score.core.types import RetrievedChunk


class LangChainRetrieverAdapter(RetrieverAdapter):
    """Wraps a langchain_core.retrievers.BaseRetriever.

    Example:
        from my_project import my_langchain_retriever
        adapter = LangChainRetrieverAdapter(my_langchain_retriever)
    """

    def __init__(self, retriever: Any) -> None:
        try:
            from langchain_core.retrievers import BaseRetriever
        except ImportError as e:
            raise ImportError(
                "langchain-core is required for LangChainRetrieverAdapter. "
                "Install it with: pip install rag-score[langchain]"
            ) from e

        if not isinstance(retriever, BaseRetriever):
            raise TypeError(
                f"Expected a langchain_core BaseRetriever, got {type(retriever).__name__}"
            )
        self._retriever = retriever

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # LangChain retrievers don't take top_k uniformly (some read it
        # from their own config, e.g. `.with_config` or constructor
        # kwargs), so we call ainvoke and truncate here rather than
        # trying to inject top_k into every possible retriever shape.
        documents = await self._retriever.ainvoke(query)
        chunks = [
            RetrievedChunk(
                # metadata key varies by vector store; doc_id/id/source
                # cover the common cases, but any consistent key works
                # as long as it matches expected_doc_ids in the dataset.
                doc_id=(
                    doc.metadata.get("doc_id")
                    or doc.metadata.get("id")
                    or doc.metadata.get("source")
                ),
                text=doc.page_content,
            )
            for doc in documents
        ]
        return chunks[:top_k]


class LangChainGeneratorAdapter(GeneratorAdapter):
    """Wraps any LangChain Runnable - an LLM, a chat model, or a full
    LCEL chain - that accepts a dict/string input and returns text via
    .ainvoke.

    If your chain expects a specific input shape (e.g. {"question":
    ..., "context": ...}), pass an input_formatter to build that dict;
    otherwise the adapter passes a plain {"question": ..., "context": ...}
    dict by default.
    """

    def __init__(self, runnable: Any, input_formatter=None) -> None:
        try:
            from langchain_core.runnables import Runnable
        except ImportError as e:
            raise ImportError(
                "langchain-core is required for LangChainGeneratorAdapter. "
                "Install it with: pip install rag-score[langchain]"
            ) from e

        if not isinstance(runnable, Runnable):
            raise TypeError(f"Expected a langchain_core Runnable, got {type(runnable).__name__}")
        self._runnable = runnable
        self._input_formatter = input_formatter

    async def generate(self, query: str, context: list[RetrievedChunk]) -> str:
        if self._input_formatter is not None:
            chain_input = self._input_formatter(query, context)
        else:
            context_text = "\n\n".join(chunk.text for chunk in context)
            chain_input = {"question": query, "context": context_text}

        output = await self._runnable.ainvoke(chain_input)
        return _extract_text(output)


def _extract_text(output: Any) -> str:
    """LangChain runnables return different shapes depending on what's
    at the end of the chain - a plain string, an AIMessage with
    .content, or a dict with an 'output'/'answer'/'text' key. Handle
    the common cases rather than forcing every user to write their own
    unwrapping code."""
    if isinstance(output, str):
        return output
    if hasattr(output, "content"):  # AIMessage, ChatMessage, etc.
        return str(output.content)
    if isinstance(output, dict):
        for key in ("answer", "output", "text", "result"):
            if key in output:
                return str(output[key])
    return str(output)
