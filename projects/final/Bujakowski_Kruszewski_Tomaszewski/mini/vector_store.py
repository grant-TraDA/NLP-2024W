from typing import List

import faiss
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from mini.pdf_utils import post_load_process_pdf


class FaissVS:
    """
    Faiss Vector Store class to create and load vector store
    """

    def __init__(self, embd_model_device: str = "cuda") -> None:
        """
        Initialize FaissVS class

        Args:
            embd_model_device (str): Device to run the embedding model. Defaults to "cuda".
        """
        self.embd_model = "BAAI/bge-base-en"
        self.embd_model_dim = 768
        self.embd_model_device = embd_model_device
        self.llm_num_output = 1024
        self.chunk_size = 256
        self.chunk_overlap = self.chunk_size * 0.1
        self.llm_context_window = 4096
        self.storage_context = None
        self.index = None

    def set_settings(self) -> None:
        """
        Set the settings for the index
        """
        embed_model = HuggingFaceEmbedding(
            model_name=self.embd_model, device=self.embd_model_device
        )
        Settings.llm = None
        Settings.embed_model = embed_model
        Settings.num_output = self.llm_num_output
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        Settings.context_window = self.llm_context_window

    def create_vector_store(self, documents: List[Document], path: str) -> None:
        """
        Create vector store from the documents

        Args:
            documents (List[Document]): List of documents
            path (str): Path to save the vector store
        """
        self.set_settings()
        faiss_index = faiss.IndexFlatL2(self.embd_model_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents=documents, storage_context=storage_context
        )
        self.index.storage_context.persist(path)

    def load_index(self, vector_store_path: str) -> "FaissVS":
        """
        Load the index from the vector store

        Args:
            vector_store_path (str): Path to the vector store

        Returns:
            FaissVS: FaissVS object
        """
        self.vector_store = FaissVectorStore.from_persist_dir(vector_store_path)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store, persist_dir=vector_store_path
        )
        self.set_settings()
        self.index = load_index_from_storage(storage_context=self.storage_context)
        return self

    def get_retriever(
        self, similarity_top_k: int, vector_store_path: str = None
    ) -> BaseRetriever:
        """
        Get retriever for the index

        Args:
            similarity_top_k (int): Number of similar documents to retrieve
            vector_store_path (str): Path to the vector store. Defaults to None.

        Returns:
            BaseRetriever: Retriever for the index
        """
        if vector_store_path is not None:
            self.load_index(vector_store_path)
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever


if __name__ == "__main__":
    faiss_vs = FaissVS()
    faiss_vs.create_vector_store(post_load_process_pdf("nlp_data.pkl"), "vector_store")
    # retriever = faiss_vs.get_retriever(2)
    # while True:
    #     query = input("Enter query: ")
    #     if query == "exit":
    #         break
        # print(retriever.retrieve(query))
