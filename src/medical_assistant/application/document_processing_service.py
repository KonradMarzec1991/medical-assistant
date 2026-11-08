from medical_assistant.application.document_classifier import DocumentClassifier, MedicalDocument
from medical_assistant.application.ports.document_extractor import DocumentExtractor


class DocumentProcessingService:
    def __init__(self, classifier: DocumentClassifier, extractor: DocumentExtractor):
        self._classifier = classifier
        self._extractor = extractor

    async def process(self, file_content: bytes, filename: str) -> MedicalDocument:
        ...
