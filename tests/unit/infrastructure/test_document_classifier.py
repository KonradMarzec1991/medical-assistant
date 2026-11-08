from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from medical_assistant.application.document_classifier import MedicalDocument, MedicalDocumentType, DocumentClassifier
from medical_assistant.application.dto.llm_response import LLMResponse
from medical_assistant.application.exceptions import LLMResponseException


async def test_document_classifier_returns_correct_classification() -> None:
    provider = AsyncMock()

    expected_content = MedicalDocument(
        is_medical_document=True,
        document_type=MedicalDocumentType.prescription,
        patient_name="joe",
        document_date=None,
    )

    provider.generate.return_value = LLMResponse(
        content=expected_content,
        input_tokens=20,
        output_tokens=30,
    )

    classifier = DocumentClassifier(
        provider=provider,
        model="test-model",
    )

    result = await classifier.classify("document-text")

    assert result == expected_content
    provider.generate.assert_awaited_once()


async def test_document_classifier_returns_incorrect_classification() -> None:
    provider = AsyncMock()

    expected_content = "some-content"

    response = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(input_tokens=20, output_tokens=30),
        output_parsed=expected_content,
    )
    parse = AsyncMock(return_value=response)
    provider.client.responses.parse = parse

    classifier = DocumentClassifier(provider=provider, model="test-model")

    with pytest.raises(LLMResponseException, match="LLM returned an unexpected response type"):
        await classifier.classify(document_text="document-text")


async def test_document_classifier_propagates_provider_exception() -> None:
    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=LLMResponseException("Provider failed"))

    classifier = DocumentClassifier(provider=provider, model="test-model")

    with pytest.raises(LLMResponseException, match="Provider failed"):
        await classifier.classify(document_text="document-text")

    provider.generate.assert_awaited_once()