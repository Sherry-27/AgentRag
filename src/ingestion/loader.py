from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_core.documents import Document
import easyocr
import numpy as np


class DocumentLoader:
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path)
        self.ocr_reader = easyocr.Reader(['en'])

    def load_pdfs(self) -> list[Document]:
        """Load all PDFs from directory"""
        loader = DirectoryLoader(
            str(self.docs_path),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} document pages")
        return docs

    def load_with_ocr(self, pdf_path: str) -> list[Document]:
        """Use EasyOCR for scanned PDFs"""
        import fitz
        pdf = fitz.open(pdf_path)
        docs = []
        for page_num, page in enumerate(pdf):
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            result = self.ocr_reader.readtext(img, detail=0)
            text = " ".join(result)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num}
                ))
        return docs