# RAPTOR

**R**ecursive **A**bstractive **P**rocessing for **T**ree-**O**rganized **R**etrieval. Build a tree
by clustering document embeddings and summarizing each cluster with an LLM; index both the original
leaves and the summaries, then retrieve over all levels at once ("collapsed tree").

```
build:  docs → embed → cluster → LLM-summarize each cluster → index {leaves + cluster summaries}
query:  dense match over all nodes → expand matched summaries to covered docs → top-k → answer
```

**Why a summary level helps:** broad/aggregation questions ("what's the overall topic", "name two
companies connected to X") match a cluster summary better than any single sentence, and the summary
node carries pointers to all its source docs. **Cost:** an LLM summary per cluster at build time.
