"""
Simple script to convert pdfs to faiss index. The pdfs are first loaded and converted to Document objects.
The documents are then split into chunks of text using the RecursiveCharacterTextSplitter.
The chunks are then converted to embeddings using the HuggingFaceInstructEmbeddings.
The embeddings are then stored in a faiss index.

Args:
    --file_path (str): Path to the folder containing pdfs
    --save_path (str): Path to save the faiss index

Example:
    python pdf2faiss.py --file_path /path/to/pdfs --save_path /path/to/save

Note:
If the save path exists, the script will try to load the index from the save path and add the new documents to the index.
"""
import argparse
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch.cuda import is_available
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL = "hkunlp/instructor-xl"
CHUNK_SIZE = 384
CHUNK_OVERLAP = 96


def load_docs(file_path: str) -> List[Document]:
    """
    Load pdfs from the given file path. Each pdf is loaded and converted to a Document object.

    Args:
        file_path (str): Path to the folder containing pdfs

    Returns:
        List[Document]: List of Document objects
    """
    pdfs = os.listdir(file_path)
    pdfs = [os.path.join(file_path, pdf) for pdf in pdfs if pdf.endswith(".pdf")]

    docs = []
    for pdf in tqdm(pdfs):
        pages = []
        loader = PyPDFLoader(pdf)
        for page in loader.load():
            pages.append(page)

        text = "\n".join(page.page_content for page in pages)
        doc = Document(page_content=text, metadata={"source": page.metadata["source"]})
        docs.append(doc)

    return docs


def split_docs(docs: List[Document], tokenizer: AutoTokenizer) -> List[Document]:
    """
    Split the documents into chunks of text using the RecursiveCharacterTextSplitter.

    Args:
        docs (List[Document]): List of Document objects
        tokenizer (AutoTokenizer): Huggingface tokenizer

    Returns:
        List[Document]: List of Document objects
    """
    docs_all = []
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    for doc in docs:
        doc_chunks = text_splitter.split_documents([doc])

        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata.update({"chunk_idx": idx})
            docs_all.append(chunk)

    return docs_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert pdfs to faiss index"
    )

    parser.add_argument(
        "--file_path", type=str, help="Path to the folder containing pdfs"
    )
    parser.add_argument("--save_path", type=str, help="Path to save the faiss index")

    args = parser.parse_args()

    file_path = args.file_path
    save_path = args.save_path

    device = "cuda" if is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=MODEL, model_kwargs={"device": device}
    )

    docs = load_docs(file_path)
    docs_all = split_docs(docs, tokenizer)

    if os.path.exists(save_path):
        try:
            db = FAISS.load_local(folder_path=save_path, index_name="faiss_index")
            db.add_documents(docs_all, embeddings)
        except:
            raise Exception("Save path exists but could not load the index")
    else:
        db = FAISS.from_documents(docs_all, embeddings)
        db.save_local(folder_path=save_path, index_name="faiss_index")
