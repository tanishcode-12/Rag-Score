# Examples

Three working, runnable end-to-end examples showing how to plug rag-score into different setups.

| File | Shows | Requires |
|---|---|---|
| `raw_example.py` | No framework at all - two plain async functions | Nothing extra |
| `langchain_example.py` | A real LangChain `BaseRetriever` + LCEL chain | `pip install rag-score[langchain]` |
| `llamaindex_example.py` | A real LlamaIndex `BaseRetriever` + bare LLM | `pip install rag-score[llamaindex]` |

Each one is self-contained and uses a tiny in-memory "document store" instead of a real vector DB, so you can run it immediately with no setup:

```bash
python examples/raw_example.py
python examples/langchain_example.py     # after installing the langchain extra
python examples/llamaindex_example.py    # after installing the llamaindex extra
```

To adapt any of them to your real pipeline, replace the demo retriever/generator classes at the top of the file with your actual vectorstore retriever and LLM call - the adapter and evaluation code below doesn't need to change.
