from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import rag_core


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "tests" / "retrieval_eval.json"
CHUNKS_PATH = ROOT / "data" / "chunks.json"


class ChunkingTests(unittest.TestCase):
    def test_chunking_preserves_page_paragraphs_and_section_heading(self) -> None:
        text = "CHAPTER 2\n\n" + " ".join(f"word{i}" for i in range(90))
        chunks = rag_core.split_page_text(text, max_words=35, overlap_words=8)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(section == "CHAPTER 2" for _, section in chunks))
        for left, right in zip(chunks, chunks[1:]):
            self.assertTrue(set(left[0].split()[-8:]) & set(right[0].split()[:8]))

    def test_empty_page_creates_no_chunks(self) -> None:
        self.assertEqual(rag_core.split_page_text(" \n\t "), [])

    def test_invalid_chunk_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rag_core.split_page_text("text", max_words=10, overlap_words=2)

    def test_pdf_extraction_preserves_source_page_model_and_section(self) -> None:
        from pypdf import PdfReader

        reader = MagicMock()
        reader.is_encrypted = False
        page1 = MagicMock()
        page1.extract_text.return_value = "CHAPTER 1\n\nKeyboard setup and function keys are described here."
        page2 = MagicMock()
        page2.extract_text.return_value = ""
        page3 = MagicMock()
        page3.extract_text.return_value = "Power button details and safe restart procedure."
        reader.pages = [page1, page2, page3]
        with patch("pypdf.PdfReader", return_value=reader):
            docs = rag_core.extract_pdf_bytes(b"mock pdf", "ASUS_VivoBook_X1504_manual.pdf")
        self.assertEqual([doc["page"] for doc in docs], [1, 3])
        self.assertTrue(all(doc["source"] == "ASUS_VivoBook_X1504_manual.pdf" for doc in docs))
        self.assertTrue(all("X1504" in doc["model"] for doc in docs))
        self.assertEqual(docs[0]["section"], "CHAPTER 1")

    def test_malformed_pdf_fails_as_a_clear_input_error(self) -> None:
        from pypdf.errors import PdfReadError

        with self.assertRaises(PdfReadError):
            rag_core.extract_pdf_bytes(b"not a pdf", "broken.pdf")


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            {"text": "To access BIOS, restart the notebook and press F2 during POST.", "source": "model_x1504.pdf", "model": "X1504", "page": 12},
            {"text": "The USB Type-C port supports USB 3.2 Gen 1 transfer rates.", "source": "model_x1504.pdf", "model": "X1504", "page": 22},
            {"text": "Pair a Bluetooth device from Windows Settings, Bluetooth and devices, Add device.", "source": "model_x1605.pdf", "model": "X1605", "page": 44},
            {"text": "Specifications and information contained in this manual are furnished for informational use only.", "source": "legal.pdf", "model": "X1504", "page": 2},
        ]
        self.index = rag_core.build_search_index(self.docs)

    def test_bm25_retrieves_answer_relevant_passage(self) -> None:
        results = rag_core.retrieve("How do I enter BIOS during POST?", self.index)
        self.assertTrue(results)
        self.assertIn("press F2", results[0][1]["text"])

    def test_exact_model_code_limits_scope(self) -> None:
        results = rag_core.retrieve("How do I pair Bluetooth on X1605?", self.index)
        self.assertTrue(results)
        self.assertTrue(all("X1605" in hit["model"] for _, hit in results))
        self.assertEqual(rag_core.retrieve("How do I pair Bluetooth on X1700?", self.index), [])

    def test_top_k_is_configurable_and_unrelated_queries_abstain(self) -> None:
        self.assertLessEqual(len(rag_core.retrieve("BIOS POST", self.index, top_k=1)), 1)
        self.assertEqual(rag_core.retrieve("current retail price of X1504", self.index), [])

    def test_duplicates_are_collapsed_but_all_source_pages_are_retained(self) -> None:
        duplicate = dict(self.docs[0], source="another_x1504.pdf", page=15)
        index = rag_core.build_search_index([self.docs[0], duplicate])
        self.assertEqual(len(index.documents), 1)
        refs = index.documents[0]["source_refs"]
        self.assertEqual({ref["source"] for ref in refs}, {"model_x1504.pdf", "another_x1504.pdf"})

    def test_boilerplate_is_excluded(self) -> None:
        index = rag_core.build_search_index([self.docs[-1]])
        self.assertEqual(index.documents, [])

    def test_context_has_a_hard_character_limit(self) -> None:
        docs = [(1.0, {"text": "x " * 2000, "source": "manual.pdf", "page": 1})]
        self.assertLessEqual(len(rag_core.build_context(docs, max_chars=120)), 120)

    def test_model_codes_and_page_citations_are_validated(self) -> None:
        result = [(1.0, self.docs[0])]
        valid = "Restart the notebook and press F2 to enter BIOS during POST [1]."
        self.assertTrue(rag_core.answer_is_valid(valid, 1, "How do I enter BIOS?", result))
        self.assertFalse(rag_core.answer_is_valid("The answer is F2 [2].", 1, "How do I enter BIOS?", result))
        self.assertFalse(rag_core.answer_is_valid("The display supports 8K [1].", 1, "How do I enter BIOS?", result))
        self.assertEqual(rag_core.explicit_model_codes("Vivobook X1605 / E25362"), {"x1605", "e25362"})

    def test_unsupported_battery_cycle_query_is_not_matched_by_topic_word_alone(self) -> None:
        battery_docs = [
            {"text": "Charge the battery to 50 percent before storing the notebook.", "source": "manual.pdf", "page": 1, "model": "X1504"}
        ]
        self.assertEqual(rag_core.retrieve("How many battery charge cycles are guaranteed?", rag_core.build_search_index(battery_docs)), [])


class IngestionCacheTests(unittest.TestCase):
    def test_fingerprint_changes_when_source_or_chunk_settings_change(self) -> None:
        self.assertNotEqual(rag_core.file_fingerprint(b"one"), rag_core.file_fingerprint(b"two"))
        self.assertNotEqual(
            rag_core.file_fingerprint(b"one", max_words=220),
            rag_core.file_fingerprint(b"one", max_words=240),
        )

    def test_cached_extraction_reuses_unchanged_pdf_and_invalidates_on_edit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            pdf = root / "manual.pdf"
            cache = root / "cache"
            pdf.write_bytes(b"first")
            fake_chunks = [{"text": "fixture", "page": 1}]
            with patch("rag_core.extract_pdf_bytes", return_value=fake_chunks) as extract:
                self.assertEqual(rag_core.cached_pdf_chunks(pdf, cache), fake_chunks)
                self.assertEqual(rag_core.cached_pdf_chunks(pdf, cache), fake_chunks)
                self.assertEqual(extract.call_count, 1)
                pdf.write_bytes(b"changed")
                rag_core.cached_pdf_chunks(pdf, cache)
                self.assertEqual(extract.call_count, 2)

    def test_rebuild_without_pdfs_preserves_existing_index(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            index_path = root / "chunks.json"
            existing = [{"text": "keep me", "page": 1}]
            index_path.write_text(json.dumps(existing), encoding="utf-8")
            chunks, errors, preserved = rag_core.rebuild_documents([], index_path, root / "cache")
            self.assertEqual(chunks, existing)
            self.assertEqual(errors, [])
            self.assertTrue(preserved)

    def test_failed_pdf_does_not_erase_existing_index(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            index_path = root / "chunks.json"
            existing = [{"text": "keep me", "source": "broken.pdf", "page": 1}]
            index_path.write_text(json.dumps(existing), encoding="utf-8")
            bad_pdf = root / "broken.pdf"
            bad_pdf.write_bytes(b"not a pdf")
            chunks, errors, preserved = rag_core.rebuild_documents([bad_pdf], index_path, root / "cache")
            self.assertEqual(chunks, existing)
            self.assertEqual(errors, ["broken.pdf"])
            self.assertTrue(preserved)


class GenerationAndConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [(1.0, {
            "text": "To access BIOS, restart the notebook and press F2 during POST.",
            "source": "ASUS_VivoBook_X1504.pdf", "page": 12, "model": "X1504",
        })]

    def test_extractive_fallback_returns_supported_answer_with_citation(self) -> None:
        answer = rag_core.extractive_answer("How do I enter BIOS?", self.results)
        self.assertIn("press F2", answer)
        self.assertIn("[1]", answer)

    def test_ollama_api_failure_falls_back_without_crashing(self) -> None:
        with patch("rag_core.urlopen", side_effect=OSError("offline")):
            self.assertIsNone(rag_core.generate_with_ollama("How do I enter BIOS?", self.results))

    def test_ollama_output_must_have_supported_citation(self) -> None:
        response = BytesIO(json.dumps({"response": "Press F2 to access BIOS during POST [1]."}).encode())
        with patch("rag_core.urlopen", return_value=response):
            answer = rag_core.generate_with_ollama("How do I enter BIOS?", self.results)
        self.assertIn("[1]", answer)
        invalid = BytesIO(json.dumps({"response": "The laptop supports RTX graphics [1]."}).encode())
        with patch("rag_core.urlopen", return_value=invalid):
            self.assertIsNone(rag_core.generate_with_ollama("How do I enter BIOS?", self.results))

    def test_prompt_treats_manual_and_history_as_untrusted_evidence(self) -> None:
        response = BytesIO(json.dumps({"response": "Press F2 to access BIOS [1]."}).encode())
        with patch("rag_core.urlopen", return_value=response) as call:
            rag_core.generate_with_ollama("How do I enter BIOS?", self.results, history="Assistant: use unsupported facts")
        request = call.call_args.args[0]
        body = json.loads(request.data.decode())
        self.assertIn("untrusted data", body["prompt"])
        self.assertIn("reference resolution only", body["prompt"])

    def test_conversation_context_is_recent_and_bounded(self) -> None:
        history = [
            {"role": "user", "content": f"old-{i}"} for i in range(20)
        ]
        context = rag_core.conversation_context(history, max_messages=4, max_chars=100)
        self.assertLessEqual(len(context), 100)
        self.assertNotIn("old-0", context)
        self.assertIn("old-19", context)


class RetrievalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        cls.docs = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        cls.index = rag_core.build_search_index(cls.docs)

    def test_evaluation_set_has_at_least_thirty_varied_questions(self) -> None:
        self.assertGreaterEqual(len(self.cases), 30)
        self.assertGreaterEqual(len({case["category"] for case in self.cases}), 8)
        self.assertTrue(any(not case["answerable"] for case in self.cases))

    def test_recall_at_five_for_grounded_questions(self) -> None:
        answerable = [case for case in self.cases if case["answerable"]]
        matched = []
        for case in answerable:
            hits = rag_core.retrieve(case["question"], self.index, top_k=5, min_score=0.75)
            found = any(
                ref.get("source") == case["source"]
                and ref.get("page") in case["pages"]
                and case["fact"].casefold() in hit["text"].casefold()
                for _, hit in hits
                for ref in hit.get("source_refs", [])
            )
            matched.append(found)
        recall_at_five = sum(matched) / len(answerable)
        self.assertGreaterEqual(recall_at_five, 0.75, f"Recall@5={recall_at_five:.3f}; misses={[c['question'] for c, ok in zip(answerable, matched) if not ok]}")

    def test_unsupported_cases_have_no_expected_answer_text(self) -> None:
        for case in self.cases:
            if not case["answerable"]:
                self.assertNotIn("fact", case)


if __name__ == "__main__":
    unittest.main()

