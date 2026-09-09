"""Tests for adapters/base.py - the zero-lock-in boundary."""

from __future__ import annotations

from rag_score.adapters.base import CallableGeneratorAdapter, CallableRetrieverAdapter
from rag_score.core.types import RetrievedChunk


class TestCallableRetrieverAdapter:
    async def test_wraps_async_function(self):
        async def retrieve(query: str, top_k: int):
            return [RetrievedChunk(doc_id="doc_1", text="x")]

        adapter = CallableRetrieverAdapter(retrieve)
        result = await adapter.retrieve("query", top_k=5)
        assert result == [RetrievedChunk(doc_id="doc_1", text="x")]

    async def test_wraps_sync_function(self):
        def retrieve(query: str, top_k: int):
            return [RetrievedChunk(doc_id="doc_2", text="y")]

        adapter = CallableRetrieverAdapter(retrieve)
        result = await adapter.retrieve("query", top_k=5)
        assert result == [RetrievedChunk(doc_id="doc_2", text="y")]

    async def test_passes_through_top_k(self):
        received = {}

        async def retrieve(query: str, top_k: int):
            received["top_k"] = top_k
            return []

        adapter = CallableRetrieverAdapter(retrieve)
        await adapter.retrieve("q", top_k=42)
        assert received["top_k"] == 42


class TestCallableGeneratorAdapter:
    async def test_wraps_async_function(self):
        async def generate(query: str, context: list[RetrievedChunk]):
            return f"answer with {len(context)} chunks"

        adapter = CallableGeneratorAdapter(generate)
        result = await adapter.generate("q", [RetrievedChunk(doc_id="d", text="t")])
        assert result == "answer with 1 chunks"

    async def test_wraps_sync_function(self):
        def generate(query: str, context: list[RetrievedChunk]):
            return "sync answer"

        adapter = CallableGeneratorAdapter(generate)
        result = await adapter.generate("q", [])
        assert result == "sync answer"
