import os
from typing import List
from uuid import uuid4

import chromadb
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Choose the model you want to use
# MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MODEL_NAME = "jinaai/jina-embeddings-v3"

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


if __name__ == "__main__":
    db_directory = DB_DIRECTORY
    processed_data_dir = "data/processed"
    collection_name = "mini-rag-bot"

    # embedder = SentenceTransformerWrapper("all-MiniLM-L6-v2")
    embedder = SentenceTransformerWrapperAPI()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chroma_client = chromadb.PersistentClient(path=db_directory)
    collection = chroma_client.get_or_create_collection(collection_name)

    for root, _, files in os.walk(processed_data_dir):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)
                print(f"Processing {input_path}")

                loader = TextLoader(input_path)
                document = loader.load()
                assert len(document) == 1
                chunks = text_splitter.split_documents(document)
                print(f"Split into {len(chunks)} chunks")
                chunks = [c.page_content for c in chunks]

                if len(chunks) == 0:
                    continue

                embeddings = embedder.embed_documents(chunks)
                collection.add(
                    documents=chunks,
                    metadatas=[{"source": input_path}] * len(chunks),
                    embeddings=embeddings,
                    ids=[str(uuid4()) for _ in range(len(chunks))],
                )
    print(
        f"Succesfully added {collection.count()} documents to collection {collection_name}"
    )

    # create vector store for retrieval
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=db_directory,
    )
