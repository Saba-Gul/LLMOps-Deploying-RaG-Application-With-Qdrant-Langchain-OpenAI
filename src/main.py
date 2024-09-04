import os
import sys
sys.path.append("C:\\Users\\Saba Gul\\Downloads\\LLMOps-main\\LLMOps-main")

from src.preprocess import load_documents, split_documents
from src.index import store_documents_to_qdrant
from langchain_openai import OpenAIEmbeddings
from src.retriever import retrieve_answer_from_docs
from src.utils import format_docs
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def retriever(question:str):
    # Load and preprocess
    # file_path = ""
    # documents = load_documents(file_path)
    # texts = split_documents(documents)
    
    # Index
    # qdrant = store_documents_to_qdrant(texts)

    
    # Retrieve
    #question = "Who was Merium?"
    answer = retrieve_answer_from_docs(question)
    
    return answer

#if __name__ == "__main__":
#    answer = retriever("Who was Merium?")
#    print(answer)
