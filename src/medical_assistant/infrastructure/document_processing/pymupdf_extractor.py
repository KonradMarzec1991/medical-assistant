import asyncio

import pymupdf


class DocumentExtractionException(Exception):
    pass


class ScannedDocumentException(DocumentExtractionException):
    pass


class PyMuPDFDocumentExtractor:
    def _extract_sync(self, file_content: bytes) -> str:
        with pymupdf.open(stream=file_content, filetype="pdf") as doc:
            pages = [
                page.get_text()
                for page in doc
            ]

        extracted = "\n".join(pages)
        if not extracted.strip():
            raise ScannedDocumentException(
                "PDF contains no extractable text. OCR is required."
            )

        return extracted

    async def extract(self, file_content: bytes, filename: str) -> str:
        return await asyncio.to_thread(
            self._extract_sync,
            file_content,
        )