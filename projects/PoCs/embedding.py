from langchain_openai import OpenAIEmbeddings
import numpy as np


def embedding(texts: list[str]) -> np.ndarray:
    return np.array([OpenAIEmbeddings(
        api_key="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"
    ).embed_query(text) for text in texts])

