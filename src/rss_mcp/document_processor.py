from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Handles loading and chunking of documents.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the processor with a text splitter.

        Args:
            chunk_size: The maximum size of a chunk.
            chunk_overlap: The overlap between consecutive chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def process_file(self, file_path: str) -> List[Document]:
        """
        Loads a markdown file and splits it into chunks.

        Args:
            file_path: The path to the markdown file.

        Returns:
            A list of Document objects, where each document is a chunk.
        """
        print(f"Processing file: {file_path}")
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()

        chunks = self.text_splitter.split_documents(documents)
        print(f"Split file into {len(chunks)} chunks.")
        return chunks
