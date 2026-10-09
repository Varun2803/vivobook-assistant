# VivoBook Manual Assistant

A Streamlit RAG assistant for ASUS VivoBook manuals. It extracts PDF text by page, splits at paragraph and section boundaries, indexes passages with BM25, and produces answers with page citations. Local Ollama generation is optional; without it, the app returns a concise extractive answer grounded in retrieved passages.

## Audit summary and architecture

The existing app already used manual passages and lexical search, so this update preserves that architecture rather than adding an unmeasured embedding model or vector database. It was not a neural embedding pipeline: it is a genuine lexical retrieval-augmented assistant. The former code rebuilt BM25 corpus statistics on every query, allowed weak topic-only matches, showed no sources by default, stored browser uploads in a shared local folder, and could overwrite the bundle with an empty index after a failed rebuild. Exact duplicate passages were also repeated across manuals.

The current flow is PDF bytes → per-page text extraction and section-aware chunks → session-cached BM25 index → model-code filtering, relevance checks, deduplication → bounded retrieved context → optional Ollama synthesis or extractive answer → citations with expandable source pages.

### What changed

- `rag_core.py` contains PDF extraction, page/section metadata, 220-word chunks with 40-word overlap, content/config/source fingerprints, cached extraction, safe index writes, duplicate collapse, BM25 ranking, model-code filters, bounded prompt context, sentence-scoped lexical citation checks, and Ollama fallback handling.
- BM25 tokenization and document frequencies are built once per Streamlit session/index version, not for each question. Exact duplicate passages are collapsed while their distinct source/page references remain available.
- Retrieval has configurable top-k, minimum score, and BM25 `k1`/`b`; it uses direct-term coverage checks, model-code scoping across retained duplicate references, query aliases, and boilerplate filtering. The BM25 parameter sweep favored `b=0` for this fixed-size passage corpus. No embedding model, FAISS, Chroma, or reranker is installed.
- PDF extraction preserves source filename, physical page number, model identifier, and a detected section heading. Empty pages are skipped; page-level extraction errors do not discard other pages. Ingestion is cached by PDF SHA-256, source filename, and chunker settings so metadata cannot leak across same-content files. Failed, empty, or absent PDFs do not evict an existing manual; unrelated manuals are retained on incremental rebuilds.
- When all families are in scope, hardware-specific wireless questions now ask for a model selection instead of mixing manual evidence. Generated-answer validation checks each cited sentence separately and rejects unsupported numeric/model identifiers; this is a conservative lexical check, not a semantic entailment guarantee.
- Browser uploads are bounded to 25 MB per file and 50 MB per session and remain only in that Streamlit session. They are not persisted or put in a process-global cache.
- Chat history is capped, only recent turns are passed as reference-resolution context, and history is explicitly not evidence. Answers retain citations and show document/page details in the expandable Sources control.
- User-facing generation errors are bounded and fall back to extracted manual text. No secrets are embedded in source.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Optional settings (environment variables or Streamlit secrets):

| Name | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama-compatible generation endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Installed model name |
| `OLLAMA_TIMEOUT_SEC` | `3` | Generation request timeout |
| `RAG_TOP_K` | `5` | Retrieved passages, bounded to 1–10 |
| `RAG_MIN_SCORE` | `1.0` | Minimum lexical relevance score |
| `RAG_BM25_K1` | `1.5` | BM25 term-frequency saturation |
| `RAG_BM25_B` | `0.0` | BM25 passage-length normalization (bounded to 0–1) |

Install Ollama and pull the selected model to enable local generation. A hosted Streamlit Cloud app cannot access your computer’s localhost; configure a reachable compatible endpoint only if you want generation in the hosted app. Manual retrieval and extractive answers work without credentials.

## Tests and evaluation

Run the offline suite with:

```powershell
python -m unittest discover -s tests -v
```

`tests/retrieval_eval.json` contains 35 questions (30 answerable, 5 intentionally unsupported) and 32 source/page/fact passage labels. `tests/evaluate_retrieval.py` reports Recall@1/3/5, MRR, citation text checks, unsupported-query results, and warm latency. Recall@k is macro-averaged per question as retrieved labelled passages divided by all labelled passages for that question; MRR uses the first relevant passage rank. Timings exclude index construction and model/API latency.

On the revised, page-checked labels, the repository's prior retrieval implementation scored R@1 0.567, R@3 0.800, R@5 0.900, and MRR 0.694. The current BM25 implementation scored 0.800, 0.933, 1.000, and 0.874 respectively. Five unsupported questions returned no passages. All 57 cited extractive sentences in the current evaluation matched text in their cited passage. See [the detailed audit and evaluation report](tests/retrieval_report.md).

The earlier 0.80 figure used a binary question-level check with a different minimum score and several faulty gold labels. We corrected three expected text snippets and replaced one unsupported “image sticking” premise with a question that the OLED manual actually answers. Baseline and current comparisons above use the same corrected 30 questions and 32 passage labels. This remains a small hand-labelled set, not a broad human correctness study. Citation matching does not establish faithfulness of Ollama-generated prose, which was not evaluated with a live model. Embedding tests are not applicable because this project has no embedding model.

## Streamlit Community Cloud deployment

1. Push the repository to GitHub with `app.py`, `rag_core.py`, `requirements.txt`, `.streamlit/config.toml`, `data/laptops.json`, and `data/packed_chunks/`.
2. In Streamlit Community Cloud, create an app from the repository and set `app.py` as the entrypoint.
3. No secret is required for manual retrieval. If using a hosted Ollama-compatible service, add `OLLAMA_HOST` and `OLLAMA_MODEL` through the app’s Secrets settings; do not commit `secrets.toml`.
4. After deploy, share the `*.streamlit.app` URL. `127.0.0.1` works only on the local computer.

### Persistence and limits

Community Cloud local files may be replaced on restart/redeploy. Keep the bundled packed index in the repository; do not rely on runtime local writes for persistent manuals. User-uploaded manuals are session-scoped and disappear when that session ends. The bundled manuals cover seven families, not every VivoBook generation. Scanned/image-only PDFs require OCR, which is not included. Hardware specifications vary by full model/SKU. The project has not had penetration testing and should not be described as production-secure.

