import asyncio

import pymupdf


class DocumentExtractionException(Exception):
    pass


class ScannedDocumentException(DocumentExtractionException):
    pass


class PyMuPDFDocumentExtractor:
    async def extract(self, file_content: bytes, filename: str) -> str:
        return await asyncio.to_thread(
            self._extract_sync,
            file_content,
        )

    def _extract_sync(self, file_content: bytes) -> str:
        try:
            with pymupdf.open(stream=file_content, filetype="pdf") as doc:
                pages = [page.get_text() for page in doc]

        except pymupdf.FileDataError as e:
            raise DocumentExtractionException("Unable to read PDF file") from e

        extracted = "\n".join(pages).strip()

        if not extracted:
            raise ScannedDocumentException(
                "PDF contains no extractable text. OCR may be required."
            )

        return extracted