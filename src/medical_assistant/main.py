import hashlib
from uuid import uuid4
from typing import Annotated

from fastapi import Depends, FastAPI, status, HTTPException, Header
from pydantic import BaseModel, Field, ConfigDict

from medical_assistant.api.dependencies import get_assistant_service, get_idempotency_service
from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException, IdempotencyReservationException, \
    IdempotencyConflictException, IdempotencyInProgressException
from medical_assistant.application.idempotency_service import IdempotencyService

app = FastAPI(title="Enterprise Medical Assistant")


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/assistant/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    service: AssistantService = Depends(get_assistant_service),
    idempotency_service: IdempotencyService = Depends(get_idempotency_service),
) -> AskResponse:
    try:
        answer = idempotency_service.execute(
            key=idempotency_key or str(uuid4()),
            request_hash=hashlib.sha256(
                request.question.encode("utf-8")
            ).hexdigest(),
            operation=lambda: service.answer(request.question),
        )
        return AskResponse(answer=answer)
    except LLMProviderException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable",
        )
    except IdempotencyConflictException:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency key was already used with a different request",
        )
    except IdempotencyInProgressException:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Request with this idempotency key is still processing",
        )
    except IdempotencyReservationException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reserve idempotency key",
        )
