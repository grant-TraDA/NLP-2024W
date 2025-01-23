from typing import List

from llama_index.core.schema import MetadataMode, NodeWithScore
from model import Llama3Generator, load_model_and_tokenizer, settings
from vector_store import FaissVS


class MINI:
    """
    Class for proviiding RAG based on MiNI lecture notes

    """

    def __init__(
        self,
        vector_store_pth: str = "vector_store",
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        """
        Constructor for MINI class

        Args:
            vector_store_pth (str): Path to the vector store
            model_name (str): Model name to be used for
        """
        self.vector_store = FaissVS()
        self.model, self.tokenizer = load_model_and_tokenizer(model_name=model_name)
        self.llm_settings = settings()
        self.generator = Llama3Generator(self.model, self.tokenizer)
        self.retriever = self.vector_store.get_retriever(
            similarity_top_k=2, vector_store_path=vector_store_pth
        )

    def change_k_similarity(self, k: int):
        """
        Change the value of k for similarity_top_k

        Args:
            k (int): Number of top k similar documents to be retrieved
        """
        self.retriever = self.vector_store.get_retriever(similarity_top_k=k)

    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer based on the prompt

        Args:
            prompt (str): Prompt for generating the answer

        Returns:
            str: Generated answer
        """
        response = self.generator.generate_simple(
            prompt,
            self.llm_settings,
            completion_only=True,
        )
        return response

    def format_sources(self, context: List[NodeWithScore]) -> List[dict]:
        """
        Format the sources in the context

        Args:
            context (List[NodeWithScore]): List of nodes with scores

        Returns:
            List[dict]: List of formatted sources
        """
        return [
            {
                **source.metadata,
                "score": f"{source.score:.3f}",
                "excerpt": source.text,
            }
            for source in context
        ]

    def format_context(self, context: List[NodeWithScore]) -> str:
        """
        Format the context

        Args:
            context (List[NodeWithScore]): List of nodes with scores

        Returns:
            str: Formatted context
        """
        return "\n".join(
            f"Excerpt {i}:\n{node.get_content(metadata_mode=MetadataMode.LLM).strip()}\n"
            for i, node in enumerate(context, 1)
        )

    def query(self, question: str) -> tuple:
        """
        Query the model for the given question

        Args:
            question (str): Question to be queried

        Returns:
            tuple: Tuple containing the answer and the sources
        """

        context = self.retriever.retrieve(question)
        context_text = self.format_context(context)
        sources = self.format_sources(context)
        query_text = f"""
        Based solely on the context provided, generate an accurate response to the question. Use only information contained in the context, not prior knowledge.
        When you are unsure about some fact, you can refer the user to check the excerpt manually.
        If the answer cannot be determined from the provided context, clearly state that the information is not available.

        CONTEXT:
        {context_text}\n\n
        QUESTION: {question}""".strip()

        answer = self.generate_answer(query_text)
        return (answer, sources)
