# Chunking-strategy RAG

Same dense retriever — different indexing granularity. *What you index* and *what you return* are
separate decisions, and both matter as much as the retriever:

| chunker | index (matched on) | return (sent to LLM) | idea |
|---|---|---|---|
| `sentence` | a sentence | that sentence | baseline granularity |
| `sentence_window` | a sentence | sentence ± neighbours | precise match, fuller context |
| `parent_child` | a sentence | the whole parent doc | small-to-big |
| `contextual` | LLM-context blurb **+** sentence | the sentence | Anthropic contextual retrieval |

```
query → dense over <chunker> index → return_text of top-k → context → answer
```

**Contextual retrieval** prepends a one-line LLM summary of the document to each chunk *before*
embedding, so an isolated sentence ("She advises Veyra on compiler design") still carries who "she"
is — fixing the classic chunk-loses-its-context failure.
