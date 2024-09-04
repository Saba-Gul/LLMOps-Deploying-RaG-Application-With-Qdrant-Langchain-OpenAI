import os
import sys
sys.path.append("C:\\Users\\Saba Gul\\Downloads\\LLMOps-main\\LLMOps-main")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from custom_logging import logger
from custom_exception import CustomException

def load_documents(file_path: str):
    """
    Load documents from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: List of documents.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info("Documents loaded successfully from %s", file_path)
        return documents
    except Exception as e:
        raise CustomException(e, sys)

def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks.

    Args:
        documents (list): List of documents.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        logger.info("Documents split into chunks successfully")
        return texts
    except Exception as e:
        raise CustomException(e, sys)
