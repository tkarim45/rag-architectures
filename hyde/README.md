# HyDE (Hypothetical Document Embeddings)

Embed a hypothetical *answer*, not the question. The LLM drafts a short passage that *would* answer
the query; you embed that draft and search. Answer-shaped text sits near real answer passages in
embedding space, so this bridges the query→document vocabulary gap.

```
query → LLM drafts a hypothetical answer → embed the draft → dense retrieve → top-k → real answer
```

**Caveat:** if the model invents an off-topic or wrong passage, you search the wrong neighborhood —
HyDE sometimes *underperforms* plain dense retrieval. This repo measures whether it helps here.
