import pytest
from fastapi.testclient import TestClient

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.main import app


client = TestClient(app)


def test_ask_request(assistant_service: AssistantService) -> None:
    response = client.post("/api/assistant/ask", json={"question": "How are you?"})

    assert response.status_code == 200
    assert response.json() == {"Testowa odpowiedź na: How are you?"}
