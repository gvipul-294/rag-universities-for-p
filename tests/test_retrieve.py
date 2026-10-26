import dataclasses
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from phd_rag.ingest import CHUNKS_FILE, INDEX_FILE
from phd_rag.retrieve import answer, search
from phd_rag.settings import get_settings


class _FakeIndex:
    ntotal = 10

    def __init__(self) -> None:
        self.requested_k = 0

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self.requested_k = k
        return np.arange(k, 0, -1, dtype="float32").reshape(1, -1), np.arange(k).reshape(1, -1)


class RetrievalCountryFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        (self.artifacts / INDEX_FILE).touch()
        records = [
            {
                "university": f"University {number}",
                "country": "Canada" if number < 9 else "Spain",
                "url": f"https://example.edu/{number}",
                "text": "Official doctoral research information.",
                "fetched_at": "2026-10-26T00:00:00+00:00",
            }
            for number in range(10)
        ]
        (self.artifacts / CHUNKS_FILE).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        self.settings = dataclasses.replace(get_settings(), root=self.root)
        self.index = _FakeIndex()

    def test_country_filter_searches_entire_index_and_returns_only_matching_country(self) -> None:
        with (
            patch("phd_rag.retrieve.faiss.read_index", return_value=self.index),
            patch("phd_rag.retrieve.embed_query", return_value=np.zeros((1, 2), dtype="float32")),
        ):
            results = search(self.settings, "doctoral research", top_k=1, countries=["Spain"])

        self.assertEqual(self.index.requested_k, self.index.ntotal)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record["country"], "Spain")

    def test_country_filter_never_falls_back_to_other_countries(self) -> None:
        with (
            patch("phd_rag.retrieve.faiss.read_index", return_value=self.index),
            patch("phd_rag.retrieve.embed_query", return_value=np.zeros((1, 2), dtype="float32")),
        ):
            results = search(self.settings, "doctoral research", top_k=2, countries=["South Korea"])

        self.assertEqual(results, [])

    def test_answer_does_not_call_language_model_when_no_sources_match(self) -> None:
        with (
            patch("phd_rag.retrieve.search", return_value=[]),
            patch("phd_rag.retrieve.OpenAI") as openai_client,
        ):
            response, results = answer(self.settings, "doctoral research", countries=["South Korea"])

        self.assertIn("No indexed university sources matched", response)
        self.assertEqual(results, [])
        openai_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
