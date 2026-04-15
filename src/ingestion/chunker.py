from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



class SemanticChunker:
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
            length_function=len,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_text"] = chunk.page_content[:100]
        print(f"Created {len(chunks)} chunks")
        return chunks