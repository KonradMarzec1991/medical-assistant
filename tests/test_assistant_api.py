from fastapi.testclient import TestClient


def test_ask_returns_answer(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Testowa odpowiedź na: How are you?"
    }


def test_ask_rejects_blank_question(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422
