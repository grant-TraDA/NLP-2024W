import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import spacy
from typing import List, Dict, Tuple
import os
import json


class PolishRAGSystem:
    def __init__(self,
                 collection_name: str,
                 persist_directory: str = "./chroma_db",
                 model_name: str = "sdadas/st-polish-paraphrase-from-mpnet"):
        """
        Initialize the RAG system with database persistence.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to store the database
            model_name: Name of the sentence transformer model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if self.device == "cuda":
            spacy.require_gpu()

        # Initialize SpaCy
        try:
            self.nlp = spacy.load("pl_core_news_lg")
        except:
            import os
            os.system("python -m spacy download pl_core_news_lg")
            self.nlp = spacy.load("pl_core_news_lg")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Setup embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=self.device,
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Set the maximum text chunk size for SpaCy processing
        self.spacy_chunk_size = 100000  # Process 100K characters at a time

    def process_text_in_chunks(self, text: str) -> List[spacy.tokens.doc.Doc]:
        """
        Process large text by breaking it into smaller chunks for SpaCy.

        Args:
            text: Input text to process

        Returns:
            List of processed SpaCy Doc objects
        """
        # Calculate number of chunks needed
        text_len = len(text)
        num_chunks = (text_len // self.spacy_chunk_size) + 1

        docs = []
        for i in range(num_chunks):
            start_idx = i * self.spacy_chunk_size
            end_idx = min((i + 1) * self.spacy_chunk_size, text_len)

            # Find the last complete sentence in chunk if possible
            if end_idx < text_len:
                last_period = text.rfind('.', start_idx, end_idx)
                if last_period != -1:
                    end_idx = last_period + 1

            chunk = text[start_idx:end_idx]
            if chunk.strip():  # Only process non-empty chunks
                docs.append(self.nlp(chunk))

        return docs

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess Polish text, handling large documents safely.
        """
        # Process text in chunks
        docs = self.process_text_in_chunks(text.lower())

        # Combine processed chunks
        all_tokens = []
        for doc in docs:
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            all_tokens.extend(tokens)

        return " ".join(all_tokens)

    def chunk_document(self,
                       text: str,
                       chunk_size: int = 1000,
                       overlap: int = 200) -> List[Dict]:
        """
        Split document into chunks with metadata, handling large documents safely.

        Returns:
            List of dictionaries containing chunks and their metadata
        """
        # Process document in chunks
        docs = self.process_text_in_chunks(text)

        # Collect all sentences
        sentences = []
        for doc in docs:
            sentences.extend([sent.text.strip() for sent in doc.sents])

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_id': chunk_id,
                        'length': len(chunk_text),
                        'start_sentence': current_chunk[0],
                        'end_sentence': current_chunk[-1]
                    }
                })
                chunk_id += 1

                # Handle overlap
                overlap_text_length = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_text_length += len(sent)
                    if overlap_text_length > overlap:
                        break
                    overlap_sentences.insert(0, sent)
                current_chunk = overlap_sentences
                current_length = sum(len(sent) for sent in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'chunk_id': chunk_id,
                    'length': len(chunk_text),
                    'start_sentence': current_chunk[0],
                    'end_sentence': current_chunk[-1]
                }
            })

        return chunks

    def add_documents(self,
                      documents: List[str],
                      metadata_list: List[Dict] = None,
                      chunk_size: int = 1000):
        """
        Process and add documents to the database.
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)

        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_counter = 0

        for doc_idx, (doc, base_metadata) in enumerate(zip(documents, metadata_list)):
            # Chunk the document
            chunks = self.chunk_document(doc, chunk_size=chunk_size)

            for chunk in chunks:
                # Combine base metadata with chunk metadata
                combined_metadata = {**base_metadata, **chunk['metadata']}
                combined_metadata['document_id'] = f"doc_{doc_idx}"

                all_chunks.append(chunk['text'])
                all_metadatas.append(combined_metadata)
                all_ids.append(f"chunk_{doc_idx}_{chunk['metadata']['chunk_id']}")

                chunk_counter += 1

                if chunk_counter >= 500:
                    # Add to ChromaDB
                    self.collection.add(
                        documents=all_chunks,
                        metadatas=all_metadatas,
                        ids=all_ids
                    )

                    all_chunks = []
                    all_metadatas = []
                    all_ids = []
                    chunk_counter = 0

        # Add any remaining chunks to ChromaDB
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )

    def search(self,
               query: str,
               top_k: int = 5,
               include_metadata: bool = True) -> List[Dict]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query in Polish
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results

        Returns:
            List of dictionaries containing matches and their metadata
        """
        # Process query
        processed_query = self.preprocess_text(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_texts=[processed_query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        formatted_results = []
        for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        ):
            result = {
                'text': doc,
                'similarity_score': 1 - distance  # Convert distance to similarity score
            }
            if include_metadata:
                result['metadata'] = metadata
            formatted_results.append(result)

        return formatted_results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the current collection."""
        return {
            'count': self.collection.count(),
            'name': self.collection.name
        }

    def delete_collection(self):
        """Delete the current collection."""
        self.chroma_client.delete_collection(self.collection.name)
