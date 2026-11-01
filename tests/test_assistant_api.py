from fastapi import status
from fastapi.testclient import TestClient


def test_ask_returns_answer(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "answer": "Testowa odpowiedź na: How are you?"
    }


def test_ask_rejects_blank_question(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "   "},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_ask_returns_503_when_provider_fails(failing_client: TestClient) -> None:
    response = failing_client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json() == {
        "detail": "AI service is temporarily unavailable"
    }
