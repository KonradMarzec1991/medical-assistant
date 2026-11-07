import hashlib
from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Annotated

from fastapi import Depends, FastAPI, status, HTTPException, Header
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict
from redis.asyncio import Redis

from medical_assistant.api.dependencies import get_assistant_service, get_idempotency_service
from medical_assistant.application.assistant_service import AssistantService
from medical_assistant.application.exceptions import LLMProviderException, IdempotencyReservationException, \
    IdempotencyConflictException, IdempotencyInProgressException
from medical_assistant.application.idempotency_service import IdempotencyService
from medical_assistant.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.open_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
    )
    app.state.redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    yield

    await app.state.openai_client.close()
    await app.state.redis_client.aclose()


app = FastAPI(title="Enterprise Medical Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/assistant/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    service: AssistantService = Depends(get_assistant_service),
    idempotency_service: IdempotencyService = Depends(get_idempotency_service),
) -> AskResponse:
    try:
        answer = await idempotency_service.execute(
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
