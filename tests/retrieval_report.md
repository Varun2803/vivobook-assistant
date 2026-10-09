# VivoBook retrieval audit and evaluation

## Audit findings

The app uses lexical BM25 retrieval with a weighted term list and hand-authored lexical aliases. It does **not** use semantic embeddings or a vector database. `app.py` builds a per-session search index, applies model-code and relevance filtering, and passes only the top passages to optional Ollama generation or the extractive fallback.

The earlier test calculated Recall@5 as a binary question hit: for each of 30 answerable questions, run `retrieve(top_k=5, min_score=0.75)` and count the question as correct if any result has the expected source, an expected page, and the expected fact as a case-insensitive substring. The score was `hits / 30`; 24 questions passed, producing 0.80. That check did not report Recall@1/3 or MRR, and its score did not use the app's default minimum score.

Auditing the labels found three expected text strings that did not occur as written in their cited passages (File Explorer, airplane mode, and liquid precautions). One OLED question asked about image sticking although the cited manual instead describes brightness and taskbar settings. Those annotation/query defects were corrected before comparing old and new retrieval, and all 32 final source/page/fact labels were checked against the bundled chunks. The revised set still has 30 answerable questions and five unsupported questions. Two questions label two passages each for procedures that span pages.

| Failure class | Finding |
|---|---|
| PDF extraction | No corrected gold fact was missing from the bundled page text. The three wording defects were evaluation labels, not extraction loss. |
| Chunking | Corrected target facts remained on their source pages; no chunk-boundary loss caused the measured misses. |
| Source/page metadata | The corrected labels validate against `data/chunks.json`. Exact duplicates retain source references. |
| Model-code filtering | It previously checked only the first metadata record on a collapsed duplicate. A passage shared by manuals could be incorrectly excluded or cite the wrong model. Filtering and returned citations now inspect and narrow all source references. |
| BM25/relevance | `precautions`, `listed`, and other request words consumed the direct-term coverage budget; model codes did too. This excluded battery passages. `recharged` and `unused` also stemmed to forms that did not match `recharge` and `using`. |
| Semantic retrieval | Not present, so there were no embedding-related failures to classify. |
| Duplicate removal | Exact text duplicates were collapsed correctly, but model scoping and citations did not use all retained references. Fixed and covered by a cross-model duplicate test. |
| Answer/citations | Ollama was not available for an end-to-end generated-answer evaluation. The extractive fallback was separately checked for exact sentence-to-citation support. |

## Changes made

- Ignore generic request words in relevance coverage and normalize `recharged`/`recharging` and `unused`/`using` consistently. Add limited lexical expansions for force-shutdown and prevention wording.
- Treat exact model codes as corpus filters, not answer evidence. Match codes against every `source_refs` record retained during duplicate collapse, and narrow result citations to the selected family.
- Tune BM25 parameters on a 25-combination grid (`k1` 0.8–2.5, `b` 0–1). `k1=1.5, b=0` gave the strongest measured rank metrics. The bundled passages are mostly fixed-size chunks with short page-tail chunks; removing length normalization helped this set.
- Make `k1` and `b` configurable as `RAG_BM25_K1` and `RAG_BM25_B`. Add `INDEX_FORMAT_VERSION` to the per-session index cache key so tokenizer changes invalidate old in-memory indexes.
- Correct the evaluation labels, include explicit multi-page ground truth, and add a repeatable evaluator and regression tests.

## Evaluation method and results

Both versions used the same 30 revised answerable questions, 32 page/fact labels, five unsupported questions, and the same 973 bundled chunks. The baseline uses the repository's pre-change `rag_core.py`; the candidate uses the current retrieval implementation. Both use top-k 5 and minimum score 1.0. Index construction is excluded from query timings. Each query was run three times against an already-built index.

Recall@k is macro-averaged over questions: for each question, divide the number of its labelled passages found in the first k results by its number of labelled passages, then average over 30 questions. MRR is the mean reciprocal rank of the first relevant passage for each question. The two-page procedure questions contribute two labelled passages each. Warm retrieval latency measures only `retrieve`; local answer latency measures retrieval plus the extractive fallback. No LLM/API call is included.

| Version | Recall@1 | Recall@3 | Recall@5 | MRR | Retrieval mean / p95 | Retrieve + extractive mean / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Previous repository implementation | 0.567 | 0.800 | 0.900 | 0.694 | 1.25 / 2.55 ms | 2.01 / 3.59 ms |
| Current implementation (`k1=1.5`, `b=0`) | 0.800 | 0.933 | 1.000 | 0.874 | 1.58 / 2.67 ms | 2.40 / 3.56 ms |

The measured mean retrieval cost rose about 0.33 ms; p95 rose about 0.12 ms. These are small local timings, so sub-millisecond differences should be treated as noisy. The rank gains are more material on this dataset: +0.233 R@1, +0.133 R@3, +0.100 R@5, and +0.181 MRR.

The prior 0.80 result is not directly comparable: it used `min_score=0.75`, binary question recall, and faulty labels. On the revised labels, the old code scored 0.90 R@5 under the current evaluation definition.

### RRF experiment

I tested Reciprocal Rank Fusion (`k=60`) over the expanded weighted-BM25 list and a direct-term BM25 list. On the same 30 questions, RRF scored R@1 0.767, R@3 0.967, R@5 1.000, and MRR 0.884. Its warm mean/p95 latency was about 3.02/4.68 ms. RRF improved R@3 and MRR slightly but lowered R@1, left R@5 unchanged, and roughly doubled retrieval latency. It was not adopted. It fused two lexical rankings; it was not semantic search.

## Unsupported questions and citations

All five deliberately unsupported questions returned no retrieved passages. The extractive fallback produced 57 citations on the 30 answerable questions; all 57 numbered citations pointed to the corresponding retrieved passage and the cited sentence appeared in that passage (57/57, 100% for this deterministic check). This is a citation/evidence check, not an independent semantic faithfulness judgment. Ollama generation was not evaluated against a live model, and no claim is made about its end-to-end answer accuracy or latency.

## Regression tests

`python -m unittest discover -s tests -v` covers PDF extraction metadata, empty/malformed input, page-aware chunking, cache fingerprints and invalidation, empty/failed rebuilds, BM25 relevance, model-code mismatch, error codes, cross-model duplicates, relevance-filter inclusion, two-page BIOS retrieval, unsupported queries, citation validation, Ollama failure handling, and evaluation metrics/labels.

The set is small and manually labelled. The BM25 parameter sweep is therefore a local tuning result, not a universal optimum; re-evaluate with fresh VivoBook questions before making broad accuracy claims. Streamlit Community Cloud will rebuild from GitHub; runtime caches and session uploads are not durable across restarts.

