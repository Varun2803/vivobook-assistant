"""Offline retrieval evaluation for the bundled manual set.

Recall@k is the macro-average, per question, of retrieved labelled passages /
all labelled passages. MRR uses the rank of the first labelled passage. Warm
latencies exclude index construction, file I/O, and any remote/local LLM call.
"""
from __future__ import annotations

import json
import statistics
import time
import sys
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag_core


def _passages(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "relevant_passages" in case:
        return case["relevant_passages"]
    return [{"source": case["source"], "pages": case["pages"], "fact": case["fact"]}]


def passage_found(passage: dict[str, Any], ranked: list[tuple[float, dict[str, Any]]], engine=rag_core) -> bool:
    for _, hit in ranked:
        if passage["fact"].casefold() not in hit.get("text", "").casefold():
            continue
        if any(
            ref.get("source") == passage["source"] and ref.get("page") in passage["pages"]
            for ref in hit.get("source_refs", engine._source_refs(hit))
        ):
            return True
    return False


def evaluate(
    cases: list[dict[str, Any]],
    index: rag_core.BM25Index,
    *,
    top_k: int = 5,
    min_score: float = 1.0,
    repetitions: int = 3,
    engine=rag_core,
    retrieval_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_kwargs = retrieval_kwargs or {}
    answerable = [case for case in cases if case["answerable"]]
    unsupported = [case for case in cases if not case["answerable"]]
    query_hits: list[list[bool]] = []
    reciprocal_ranks: list[float] = []
    retrieval_times: list[float] = []
    answer_times: list[float] = []
    citation_checks: list[bool] = []
    for case in answerable:
        labels = _passages(case)
        ranked = engine.retrieve(case["question"], index, top_k=50, min_score=min_score, **retrieval_kwargs)
        relevant_ranks = []
        for label in labels:
            found_rank = next((rank for rank, (_, hit) in enumerate(ranked, 1) if passage_found(label, [(0, hit)], engine)), None)
            if found_rank is not None:
                relevant_ranks.append(found_rank)
        reciprocal_ranks.append(1 / min(relevant_ranks) if relevant_ranks else 0.0)
        query_hits.append([any(
            passage_found(label, ranked[:k], engine) for label in labels
        ) for k in (1, 3, 5)])
        answer = engine.extractive_answer(case["question"], ranked[:top_k])
        # Every citation must point to a retrieved passage containing the exact
        # extractive sentence immediately before that citation.
        for claim, citation in re.findall(r"([^\n]+?)\s+\[(\d+)\]", answer):
            cite = int(citation)
            valid = 1 <= cite <= min(top_k, len(ranked))
            if valid:
                source = engine.normalize_document_text(ranked[cite - 1][1].get("text", ""))
                statement = engine.normalize_document_text(claim)
                valid = bool(statement and statement in source)
            citation_checks.append(valid)
    for _ in range(repetitions):
        for case in answerable:
            started = time.perf_counter()
            ranked = engine.retrieve(case["question"], index, top_k=top_k, min_score=min_score, **retrieval_kwargs)
            retrieval_finished = time.perf_counter()
            engine.extractive_answer(case["question"], ranked)
            finished = time.perf_counter()
            retrieval_times.append((retrieval_finished - started) * 1000)
            answer_times.append((finished - started) * 1000)
    unsupported_hits = [
        case["question"] for case in unsupported
        if engine.retrieve(case["question"], index, top_k=top_k, min_score=min_score, **retrieval_kwargs)
    ]
    sorted_retrieval = sorted(retrieval_times)
    sorted_answer = sorted(answer_times)
    passage_count = sum(len(_passages(case)) for case in answerable)
    return {
        "answerable_questions": len(answerable),
        "labelled_passages": passage_count,
        "recall_at_1": statistics.mean(row[0] for row in query_hits),
        "recall_at_3": statistics.mean(row[1] for row in query_hits),
        "recall_at_5": statistics.mean(row[2] for row in query_hits),
        "mrr": statistics.mean(reciprocal_ranks),
        "unanswerable_questions": len(unsupported),
        "unsupported_false_positive_count": len(unsupported_hits),
        "unsupported_false_positive_questions": unsupported_hits,
        "extractive_citation_correctness": statistics.mean(citation_checks) if citation_checks else None,
        "extractive_citations_checked": len(citation_checks),
        "warm_retrieval_latency_ms_mean": statistics.mean(retrieval_times),
        "warm_retrieval_latency_ms_p95": sorted_retrieval[int(0.95 * (len(sorted_retrieval) - 1))],
        "warm_retrieval_plus_extractive_latency_ms_mean": statistics.mean(answer_times),
        "warm_retrieval_plus_extractive_latency_ms_p95": sorted_answer[int(0.95 * (len(sorted_answer) - 1))],
        "retrieval_misses": [
            case["question"] for case, row in zip(answerable, query_hits) if not row[2]
        ],
    }


def main() -> None:
    docs = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "tests" / "retrieval_eval.json").read_text(encoding="utf-8"))
    index = rag_core.build_search_index(docs)
    result = evaluate(cases, index)
    result["raw_chunks"] = len(docs)
    result["deduplicated_chunks"] = len(index.documents)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


