import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from phd_rag.api import create_app
from phd_rag.retrieve import RetrievedChunk
from phd_rag.settings import Settings


def _settings(root: Path) -> Settings:
    return Settings(
        root=root,
        embedding_model="text-embedding-3-small",
        chat_model="gpt-5-mini",
        ollama_model="mistral",
        ollama_base_url="http://localhost:11434",
        agent_min_top_score=0.30,
        agent_min_unique_universities=2,
        agent_max_retrieval_attempts=2,
        cors_allow_origins=["http://localhost:3000"],
        api_bearer_token=None,
    )


class ApiTests(unittest.TestCase):
    def test_health_reports_a_missing_index_without_calling_openai(self) -> None:
        with TemporaryDirectory() as directory:
            with patch("phd_rag.api.get_settings", return_value=_settings(Path(directory))):
                response = TestClient(create_app()).get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])

    def test_ask_returns_answer_and_frontend_source_fields(self) -> None:
        source = RetrievedChunk(
            score=0.81,
            record={
                "university": "Example University",
                "country": "Canada",
                "url": "https://example.edu/phd",
                "text": "Official PhD research excerpt.",
                "fetched_at": "2026-10-21T00:00:00+00:00",
            },
        )
        with TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            with (
                patch("phd_rag.api.get_settings", return_value=settings),
                patch("phd_rag.api.answer", return_value=("Grounded answer", [source])),
            ):
                response = TestClient(create_app()).post(
                    "/api/v1/ask", json={"question": "Find Canadian AI PhDs"}
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Grounded answer")
        self.assertEqual(body["source_count"], 1)
        self.assertEqual(body["sources"][0]["university"], "Example University")


if __name__ == "__main__":
    unittest.main()
