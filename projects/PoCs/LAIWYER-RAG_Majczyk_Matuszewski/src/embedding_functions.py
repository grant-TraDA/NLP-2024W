from typing import List

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embeddings(sentences: List[str]) -> List[List[float]]:
    """
    Returns the embeddings of the input sentences.

    Args:
        sentences (List[str]): List of sentences to embed.

    Returns:
        List[List[float]]: List of embeddings of the input sentences.
    """

    return model.encode(sentences)
