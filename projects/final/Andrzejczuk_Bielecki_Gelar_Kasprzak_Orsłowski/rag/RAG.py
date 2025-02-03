import os
from pathlib import Path

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAG:
    """
    Class to represent a Retrieval Augmented Generation model
    """
    DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / 'databases' / 'books_default'

    def __init__(
        self,
        postprocessors=(clean_extra_whitespace, group_broken_paragraphs),
        chunk_max_size=2000,
        chunk_overlap=200,
        device="cuda",
        prompt="Based on these passages:\n\n {} \n\n Answer the question {}. Use 1-2 sentences if possible.",
        n_retrieved=20,
        n_used=5,
        passage_sep="\n\n",
    ) -> None:
        """
        Initialize the RAG model
        :param postprocessors: List of postprocessors to apply to the text
        :param chunk_max_size: Maximum size of the chunks to split the text into
        :param chunk_overlap: Size of the overlap between the chunks
        :param device: Device to use for the model
        :param prompt: Prompt to use for the model
        :param n_retrieved: Number of passages to retrieve
        :param n_used: Number of passages to use
        :param passage_sep: Separator to use between the passages
        """
        self.chunk_max_size = chunk_max_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.prompt = prompt
        self.books = []
        self.postprocessors = postprocessors
        self.vector_store = None
        self.n_retrieved = n_retrieved
        self.n_used = n_used
        self.passage_sep = passage_sep

    def load_database(self, path):
        """
        Load the database from a path
        :param path: Path to the database
        """
        if path:
            self.vector_store = FAISS.load_local(
                path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vector_store = None
        return self

    def load_embedding_model(
        self, name="sentence-transformers/all-MiniLM-L6-v2", **kwargs
    ):
        self.embedding_model = HuggingFaceEmbeddings(model_name=name, **kwargs)
        return self

    def load_reranker(self, name="BAAI/bge-reranker-large", **kwargs):
        self.reranker_model = CrossEncoder(model_name=name, **kwargs)
        return self

    def load_llm(self, name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)
        self.llm = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", torch_dtype="auto", **kwargs
        )
        return self

    def recreate_database(self):
        """
        Recreate the database using parameters in self
        """
        loaders = [
            UnstructuredFileLoader(
                book,
                post_processors=self.postprocessors,
                paragraph_grouper=group_broken_paragraphs,
            )
            for book in self.books
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ".", "?", "\n"],
            chunk_size=self.chunk_max_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator="end",
        )
        passages = []
        for loader in loaders:
            passages.extend(loader.load_and_split(text_splitter))

        self.vector_store = FAISS.from_documents(passages, self.embedding_model)
        return self

    def add_books(self, book_list):
        """
        Add books to the model
        :param book_list: List of books to add
        """
        self.books.extend(book_list)
        if self.vector_store:
            loaders = [
                UnstructuredFileLoader(
                    book,
                    post_processors=self.postprocessors,
                    paragraph_grouper=group_broken_paragraphs,
                )
                for book in book_list
            ]
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", ".", "?", "\n"],
                chunk_size=self.chunk_max_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                keep_separator="end",
            )
            passages = []
            for loader in loaders:
                passages.extend(loader.load_and_split(text_splitter))

            self.vector_store.add_documents(passages)
        else:
            self.recreate_database()
        return self

    def retrieve(self, query, document=None):
        """
        Retrieve a query
        :param query: Query to retrieve
        :param document: Optional document to filter on (has to be in form of path to the sorted document)
        """
        if document:
            filter = {"source": document}
        else:
            filter = {}
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.n_retrieved, "filter": filter}
        )
        return retriever.invoke(query)
    
    def rerank(self, query, passages):
        """
        Rerank the passages
        :param query: Query to rerank
        :param passages: Passages to rerank
        """
        data = [(query, passage) for passage in passages]
        scores = self.reranker_model.predict(data)
        idx_used = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.n_used
        ]
        return [passages[idx] for idx in idx_used]

    def answer(self, query, max_new_tokens=200, document=None):
        """
        Answer a query
        :param query: Query to answer
        :param max_new_tokens: Maximum number of tokens to generate
        :param document: Optional document to filter on (has to be in form of path to the sorted document)
        """
        retrieved_passages = self.retrieve(query, document)
        passages_contents = [p.page_content for p in retrieved_passages]
        final_passages = self.rerank(query, passages_contents)
        passages_str = self.passage_sep.join(final_passages)
        prompt = self.prompt.format(passages_str, query)
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.llm.generate(**input_tokens, max_new_tokens=max_new_tokens)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded_output.removeprefix(prompt)
    
    def save_vector_store(self, path):
        """
        Save the vector store to a path
        :param path: Path to save the vector store
        """
        self.vector_store.save_local(path)
        return self

    @staticmethod
    def quickstart(use_default_db=True, book_dir=None):
        """
        Quickstart the RAG model
        :param use_default_db: Whether or not to start with the database that contains default books. True by default
        :param book_dir: Directory containing the books. Leave `None` if you wish to add no books
        """
        rag = RAG().load_embedding_model().load_reranker().load_llm()
        
        if use_default_db:
            rag.load_database(RAG.DEFAULT_DB_PATH)

        if book_dir is not None:
            rag.add_books(
                [str(p) for p in Path(book_dir).glob("*") if p.suffix in [".txt", ".pdf"]]
            )

        return rag
