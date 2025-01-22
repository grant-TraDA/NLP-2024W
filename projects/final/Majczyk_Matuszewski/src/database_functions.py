import psycopg2
from typing import List
from src.embedding_functions import get_embeddings

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple
import numpy as np


def create_connection(
    dbname: str = "pgvector_nlp_db",
    user: str = "postgres",
    password: str = "yourpassword",
    host: str = "localhost",
    port: str = "56434",
) -> psycopg2.extensions.connection:
    """
    Create a connection to the database.

    Args:
        dbname (str): The name of the database.
        user (str): The username to connect with.
        password (str): The password to connect with.
        host (str): The host of the database.
        port (str): The port of the database.

    Returns:
        psycopg2.connection: The connection to the database.
    """
    conn = psycopg2.connect(
        dbname=dbname, user=user, password=password, host=host, port=port
    )
    return conn


def get_cursor(conn: psycopg2.extensions.connection) -> psycopg2.extensions.cursor:
    """
    Get a cursor from the connection.

    Args:
        conn (psycopg2.connection): The connection to the database.

    Returns:
        psycopg2.cursor: The cursor to the database.
    """
    try:
        cursor = conn.cursor()
    except ConnectionError as e:
        print(f"Error: {e}")
        raise e
    return cursor


def retrieve_related_articles(
    conn: psycopg2.extensions.connection, text: str, num_articles: int = 10
):
    """
    Retrieve the most similar articles to the given embedding.

    Args:
        conn (psycopg2.connection): The connection to the database.
        text (str): The text to find similar articles to.
        num_articles (int): The number of articles to return.

    """

    embedding = get_embeddings(text).tolist()
    cursor = get_cursor(conn)
    sql = f"""
    SELECT section, section_title, article, text,
        embedding <#> %s::vector AS similarity
    FROM constitution_embeddings
    ORDER  BY similarity 
    LIMIT {num_articles};
    """
    try:
        cursor.execute(sql, (embedding,))
        results = cursor.fetchall()
    except Exception as e:
        print(f"Error: {e}")
        raise e

    return results


def insert_embedding(
    conn: psycopg2.extensions.connection,
    section: str,
    section_title: str,
    article: str,
    text: str,
    embedding: List[float],
) -> bool:
    """
    Insert an embedding into the database.

    Args:
        conn (psycopg2.connection): The connection to the database.
        section (str): The section of the constitution.
        section_title (str): The title of the section.
        article (str): The article number.
        text (str): The text of the section.
        embedding (List[float]): The embedding to insert.
    """

    cursor = get_cursor(conn)

    sql = """
    INSERT INTO constitution_embeddings (section, section_title, article, text, embedding)
    VALUES (%s, %s, %s, %s, %s)
    """

    try:
        cursor.execute(sql, (section, section_title, article, text, embedding))

    except Exception as e:
        print(f"Error: {e}")
        raise e
    return True


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name (str): The name of the model to use for reranking.
        """

        model_name = "sdadas/polish-reranker-large-ranknet"
        print(f"Using reranker: {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

    def rerank(
        self, 
        query: str, 
        documents: List[Tuple], 
        batch_size: int = 32
    ) -> List[Tuple]:
        """
        Rerank the retrieved documents using the cross-encoder model.
        
        Args:
            query (str): The search query
            documents (List[Tuple]): List of document tuples from initial retrieval
            batch_size (int): Batch size for processing
            
        Returns:
            List[Tuple]: Reranked documents with updated similarity scores
        """
        # Prepare text pairs for reranking
        pairs = [(query, doc[3]) for doc in documents]  # doc[3] contains the text

        # Calculate scores in batches
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            # Tokenize
            features = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            features = {k: v.to(self.device) for k, v in features.items()}

            # Get scores
            with torch.no_grad():
                scores.extend(
                    self.model(**features)
                    .logits
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                    .tolist()
                )

        # Combine original documents with new scores
        scored_docs = [(doc, score) for doc, score in zip(documents, scores)]

        # Sort by new scores
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Return just the documents in new order
        return [doc for doc, _ in reranked_docs]

def retrieve_and_rerank(
    conn,
    text: str,
    num_initial_candidates: int = 20,
    num_results: int = 5
) -> List[Tuple]:
    """
    Retrieve candidates using vector similarity and then rerank them.
    
    Args:
        conn: Database connection
        text (str): Query text
        num_initial_candidates (int): Number of initial candidates to retrieve
        num_results (int): Number of final results to return
        
    Returns:
        List[Tuple]: Reranked and filtered results
    """
    # Initialize reranker
    reranker = Reranker()
    
    # Get initial candidates using vector similarity
    initial_results = retrieve_related_articles(
        conn=conn,
        text=text,
        num_articles=num_initial_candidates
    )
    
    # Rerank the results
    reranked_results = reranker.rerank(
        query=text,
        documents=initial_results
    )
    
    # Return top k results after reranking
    return reranked_results[:num_results]
