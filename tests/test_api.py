import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import openai
from fastapi.testclient import TestClient

from phd_rag.api import create_app
from phd_rag.retrieve import RetrievedChunk
from phd_rag.settings import Settings, get_settings


def _settings(root: Path, token: str | None = None) -> Settings:
    # Start from the real settings so new or renamed fields can't drift out of the tests.
    return dataclasses.replace(
        get_settings(),
        root=root,
        llm_model="test-llm-model",
        cors_allow_origins=["http://localhost:3000"],
        api_bearer_token=token,
    )


SOURCE = RetrievedChunk(
    score=0.81,
    record={
        "university": "Example University",
        "country": "Canada",
        "url": "https://example.edu/phd",
        "text": "Official PhD research excerpt.",
        "fetched_at": "2026-10-21T00:00:00+00:00",
    },
)


class ApiTests(unittest.TestCase):
    def test_health_reports_a_missing_index_without_calling_openai(self) -> None:
        with TemporaryDirectory() as directory:
            with patch("phd_rag.api.get_settings", return_value=_settings(Path(directory))):
                response = TestClient(create_app()).get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])

    def test_ask_returns_answer_and_frontend_source_fields(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            with (
                patch("phd_rag.api.get_settings", return_value=settings),
                patch("phd_rag.api.answer", return_value=("Grounded answer", [SOURCE])),
            ):
                response = TestClient(create_app()).post(
                    "/api/v1/ask", json={"question": "Find Canadian AI PhDs"}
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Grounded answer")
        self.assertEqual(body["source_count"], 1)
        self.assertEqual(body["sources"][0]["university"], "Example University")
        self.assertEqual(body["chat_model"], "test-llm-model")

    def test_match_returns_answer(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch("phd_rag.api.get_settings", return_value=_settings(Path(directory))),
                patch("phd_rag.api.answer", return_value=("Matched", [SOURCE])),
            ):
                response = TestClient(create_app()).post(
                    "/api/v1/match",
                    json={"field": "Computer Science", "degree": "BTech", "research_interests": ["NLP"]},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Matched")

    def test_ask_requires_bearer_token_when_configured(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _settings(Path(directory), token="s3cret")
            with (
                patch("phd_rag.api.get_settings", return_value=settings),
                patch("phd_rag.api.answer", return_value=("Grounded answer", [SOURCE])),
            ):
                client = TestClient(create_app())
                missing = client.post("/api/v1/ask", json={"question": "Find PhDs"})
                wrong = client.post("/api/v1/ask", json={"question": "Find PhDs"}, headers={"Authorization": "Bearer nope"})
                right = client.post("/api/v1/ask", json={"question": "Find PhDs"}, headers={"Authorization": "Bearer s3cret"})

        self.assertEqual((missing.status_code, wrong.status_code, right.status_code), (401, 401, 200))

    def test_provider_quota_error_maps_to_503(self) -> None:
        request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
        quota = openai.RateLimitError("quota", response=httpx.Response(429, request=request), body=None)
        with TemporaryDirectory() as directory:
            with (
                patch("phd_rag.api.get_settings", return_value=_settings(Path(directory))),
                patch("phd_rag.api.answer", side_effect=quota),
            ):
                response = TestClient(create_app()).post("/api/v1/ask", json={"question": "Find PhDs"})

        self.assertEqual(response.status_code, 503)

    def test_missing_llm_key_maps_to_503(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch("phd_rag.api.get_settings", return_value=_settings(Path(directory))),
                patch("phd_rag.api.answer", side_effect=RuntimeError("LLM_API_KEY is missing. Add your key.")),
            ):
                response = TestClient(create_app()).post("/api/v1/ask", json={"question": "Find PhDs"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM_API_KEY", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
