import os
import pickle as pkl
from pathlib import Path
from typing import List

from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader
from tqdm import tqdm


def post_load_process_pdf(documents_path: str) -> List[Document]:
    """
    Applies post-load processing to the documents.

    Args:
        documents_path (str): Path to the pickle file containing the documents.

    Returns:
        List[Document]: List of documents.
    """
    with open(documents_path, "rb") as file:
        documents = pkl.load(file)

    for document in documents:
        document.excluded_llm_metadata_keys = ["page_label"]
        document.excluded_embed_metadata_keys = ["page_label"]
        document.text = document.text.strip()

    return documents


def extract_pdfs(pdf_dir: str, save_pkl: str) -> List[Document]:
    """
    Extracts text from PDF files and saves it in a pickle file.

    Args:
        pdf_dir (str): Directory containing PDF files.
        save_pkl (str): Path to save the pickle file.

    Returns:
        List[Document]: List of documents.
    """
    reader = PyMuPDFReader()
    pdf_lib_path = Path(pdf_dir)

    documents = []
    for pdf_file in tqdm(os.listdir(pdf_lib_path)):
        pdf_path = pdf_lib_path / pdf_file
        if not pdf_path.is_file():
            continue

        pdf_pages = reader.load_data(pdf_path)
        for page in pdf_pages:
            page.metadata = {
                "file_name": pdf_file,
                "page_label": page.metadata.get("source", ""),
            }
            documents.append(page)

    with open(save_pkl, "wb") as file:
        pkl.dump(documents, file)

    return documents


if __name__ == "__main__":
    extract_pdfs("./nlp_data", "nlp_data.pkl")
