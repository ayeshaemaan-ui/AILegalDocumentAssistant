"""API surface: routing, validation, and error contracts.

These deliberately do NOT enter the app's lifespan. Starlette only runs startup
when TestClient is used as a context manager, and startup loads a 440MB
embedding model - which would make CI download it on every run. Constructing
the client directly exercises the routing and validation layer, which is what
these tests are about.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestSchemaContract:
    def test_openapi_is_generated(self):
        spec = app.openapi()
        assert spec["info"]["title"] == "LegalLens API"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/chat/ask",
            "/api/chat/stream",
            "/api/chat/agent",
            "/api/documents",
            "/api/documents/upload",
            "/api/documents/clause-types",
            "/api/eval/run",
            "/api/eval/runs",
            "/api/health",
            "/api/config",
            "/api/cost/summary",
            "/api/traces",
            "/api/sessions",
        ],
    )
    def test_endpoint_is_registered(self, path):
        assert path in app.openapi()["paths"]


class TestRoot:
    def test_root_carries_the_disclaimer(self):
        """The positioning as a research tool is part of the product, so it is
        asserted rather than left to documentation."""
        response = client.get("/")
        assert response.status_code == 200
        assert "not legal advice" in response.json()["disclaimer"].lower()


class TestValidation:
    def test_rejects_an_empty_question(self):
        assert client.post("/api/chat/ask", json={"question": ""}).status_code == 422

    def test_rejects_an_oversized_question(self):
        assert client.post("/api/chat/ask", json={"question": "x" * 5000}).status_code == 422

    def test_rejects_a_missing_question(self):
        assert client.post("/api/chat/ask", json={}).status_code == 422

    def test_rejects_an_unknown_agent_mode(self):
        response = client.post("/api/chat/agent", json={"question": "hi", "mode": "nonsense"})
        assert response.status_code == 422

    def test_rejects_an_out_of_range_sample_size(self):
        assert client.post("/api/eval/run", json={"sample_size": 9999}).status_code == 422

    def test_accepts_filters_on_ask(self):
        """Validation must pass even with no API key - the failure should come
        from the model call, not from schema rejection."""
        from app.api.schemas import AskRequest

        request = AskRequest(question="What is the governing law?", doc_ids=["doc1"],
                             clause_types=["governing_law"])
        assert request.filters()["doc_ids"] == ["doc1"]


class TestNotFound:
    def test_unknown_document(self):
        response = client.get("/api/documents/does-not-exist")
        assert response.status_code == 404
        assert "does-not-exist" in response.json()["detail"]

    def test_unknown_trace(self):
        assert client.get("/api/traces/nonexistent").status_code == 404

    def test_unknown_eval_run(self):
        assert client.get("/api/eval/runs/nonexistent").status_code == 404

    def test_unknown_job(self):
        assert client.get("/api/documents/jobs/nope").status_code == 404


class TestUploadGuards:
    def test_rejects_an_unsupported_file_type(self):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]


class TestRouteOrdering:
    def test_static_paths_win_over_the_catch_all(self):
        """/api/documents/stats must not be swallowed by /api/documents/{doc_id}."""
        paths = app.openapi()["paths"]
        assert "/api/documents/stats" in paths
        assert "/api/documents/clause-types" in paths
        response = client.get("/api/documents/clause-types")
        assert response.status_code == 200
        assert len(response.json()["clause_types"]) == 41
