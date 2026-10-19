import pytest

from conftest import MCP_COURSE


class TestQueryEndpoint:
    def test_returns_answer_sources_and_new_session(self, api_client, mock_rag):
        sources = [{"text": f"{MCP_COURSE} - Lesson 1", "url": "https://example.com/l1"}]
        mock_rag.query.return_value = ("MCP standardizes context.", sources)

        response = api_client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "MCP standardizes context."
        assert body["sources"] == sources
        assert mock_rag.session_manager.has_session(body["session_id"])
        mock_rag.query.assert_called_once_with("What is MCP?", body["session_id"])

    def test_source_without_url_serializes_as_null(self, api_client, mock_rag):
        mock_rag.query.return_value = ("Answer.", [{"text": MCP_COURSE}])

        body = api_client.post("/api/query", json={"query": "q"}).json()

        assert body["sources"] == [{"text": MCP_COURSE, "url": None}]

    def test_reuses_known_session(self, api_client, mock_rag):
        session_id = mock_rag.session_manager.create_session()

        body = api_client.post("/api/query", json={"query": "q", "session_id": session_id}).json()

        assert body["session_id"] == session_id
        mock_rag.query.assert_called_once_with("q", session_id)

    def test_unknown_session_gets_a_new_one(self, api_client, mock_rag):
        body = api_client.post("/api/query", json={"query": "q", "session_id": "session_999"}).json()

        assert body["session_id"] != "session_999"
        assert mock_rag.session_manager.has_session(body["session_id"])

    @pytest.mark.parametrize("payload", [{}, {"query": 42}, {"session_id": "session_1"}])
    def test_invalid_request_is_rejected(self, api_client, mock_rag, payload):
        response = api_client.post("/api/query", json=payload)

        assert response.status_code == 422
        mock_rag.query.assert_not_called()

    def test_rag_failure_returns_500_with_detail(self, api_client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("LLM unreachable")

        response = api_client.post("/api/query", json={"query": "q"})

        assert response.status_code == 500
        assert response.json() == {"detail": "LLM unreachable"}


class TestCoursesEndpoint:
    def test_returns_course_stats(self, api_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": [MCP_COURSE, "Another Course"],
        }

        response = api_client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {"total_courses": 2, "course_titles": [MCP_COURSE, "Another Course"]}

    def test_empty_catalog(self, api_client):
        assert api_client.get("/api/courses").json() == {"total_courses": 0, "course_titles": []}

    def test_analytics_failure_returns_500(self, api_client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("chroma down")

        response = api_client.get("/api/courses")

        assert response.status_code == 500
        assert response.json() == {"detail": "chroma down"}


class TestSessionEndpoint:
    def test_delete_discards_history(self, api_client, mock_rag):
        session_id = mock_rag.session_manager.create_session()

        response = api_client.delete(f"/api/session/{session_id}")

        assert response.status_code == 204
        assert not mock_rag.session_manager.has_session(session_id)

    def test_delete_unknown_session_is_a_no_op(self, api_client):
        assert api_client.delete("/api/session/session_999").status_code == 204


class TestFrontend:
    def test_root_serves_index_html(self, api_client):
        response = api_client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<title>Course Materials Assistant</title>" in response.text

    def test_static_assets_are_served(self, api_client):
        assert api_client.get("/script.js").status_code == 200

    def test_unknown_path_is_404(self, api_client):
        assert api_client.get("/does-not-exist.js").status_code == 404
