from typing import Any, Dict, List

import requests
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# Choose the model you want to use
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# MODEL_NAME = "jinaai/jina-embeddings-v3"

DB_DIRECTORY = f"data/db/{MODEL_NAME.split('/')[-1]}"
# Replace Embedding API URL with the URL of the API you deployed
EMBEDDING_MODEL_API_URL = "http://localhost:8085/embeddings"


class SentenceTransformerWrapperAPI:
    def __init__(self):
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(EMBEDDING_MODEL_API_URL, json={"messages": texts})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, query: str) -> List[float]:
        response = requests.post(EMBEDDING_MODEL_API_URL, json={"messages": query})
        response.raise_for_status()
        return response.json()["embeddings"][0]


class SentenceTransformerWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()


# embedder = SentenceTransformerWrapper("all-MiniLM-L6-v2")
embedder = SentenceTransformerWrapperAPI()

vector_store = Chroma(
    collection_name="mini-rag-bot",
    embedding_function=embedder,
    persist_directory=DB_DIRECTORY,
)


def retrieve_k_most_similar_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results = vector_store.similarity_search_by_vector(
        embedding=embedder.embed_query(query),
        k=k,
    )
    return [doc.page_content for doc in results]
