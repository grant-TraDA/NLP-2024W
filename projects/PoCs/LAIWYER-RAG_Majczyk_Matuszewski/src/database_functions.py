import psycopg2
from typing import List
from src.embedding_functions import get_embeddings


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
    conn: psycopg2.extensions.connection, text: str, num_articles: int = 5
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
