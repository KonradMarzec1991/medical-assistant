import uuid
from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.infrastructure.repositories.redis_idempotency_repository import RedisIdempotencyRepository


def test_ask_returns_answer(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
        headers={"Idempotency-Key": uuid.uuid4().hex}
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "answer": "Testowa odpowiedź na: How are you?"
    }


def test_ask_rejects_blank_question(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "   "},
        headers={"Idempotency-Key": uuid.uuid4().hex}
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_ask_returns_503_when_provider_fails(failing_client: TestClient) -> None:
    response = failing_client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
        headers={"Idempotency-Key": uuid.uuid4().hex}
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json() == {
        "detail": "AI service is temporarily unavailable"
    }


def test_ask_returns_answer_after_retries(retry_client) -> None:
    client, provider = retry_client
    response = client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
        headers={"Idempotency-Key": uuid.uuid4().hex}
    )

    assert provider.generate.call_count == 3

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "answer": "Everything is fine"
    }


def test_ask_returns_redis_values(client: TestClient) -> None:
    idempotency_key = uuid.uuid4().hex

    with patch.object(AssistantService, "answer", return_value="Testowa odpowiedź na: How are you?") as mock_answer:
        response = client.post(
            "/api/assistant/ask",
            json={"question": "How are you?"},
            headers={"Idempotency-Key": idempotency_key}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "answer": "Testowa odpowiedź na: How are you?"
        }

        response = client.post(
            "/api/assistant/ask",
            json={"question": "How are you?"},
            headers={"Idempotency-Key": idempotency_key}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "answer": "Testowa odpowiedź na: How are you?"
        }

        mock_answer.assert_called_once_with("How are you?")

        repository = RedisIdempotencyRepository()
        repository.client.delete(f"idempotency:{idempotency_key}")


def test_ask_returns_conflict_when_same_key_is_used_with_different_question(client: TestClient) -> None:
    idempotency_key = uuid.uuid4().hex

    first_response = client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
        headers={"Idempotency-Key": idempotency_key},
    )

    second_response = client.post(
        "/api/assistant/ask",
        json={"question": "What is your name?"},
        headers={"Idempotency-Key": idempotency_key},
    )

    assert first_response.status_code == status.HTTP_200_OK
    assert second_response.status_code == status.HTTP_409_CONFLICT


def test_ask_requires_idempotency_key(client: TestClient) -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "How are you?"},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT