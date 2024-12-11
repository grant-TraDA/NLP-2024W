import argparse
from pathlib import Path
from tqdm import tqdm
from database import RAGSystem


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()


def process_files(rag_system, path: Path, recursive: bool, batch_size: int):
    # Collect files to process
    if path.is_file():
        if path.suffix.lower() == '.txt':
            files = [path]
        else:
            raise ValueError(f"Not a text file: {path}")
    elif path.is_dir():
        pattern = '**/*.txt' if recursive else '*.txt'
        files = list(path.glob(pattern))
    else:
        raise ValueError(f"Invalid path: {path}")

    if not files:
        print(f"No text files found in {path}")
        return {'processed': 0, 'total': 0}

    # Process files in batches
    documents = []
    metadata_list = []
    stats = {'processed': 0, 'total': len(files)}

    for i, file_path in enumerate(tqdm(files, desc="Processing files")):
        try:
            content = read_text_file(str(file_path))
            metadata = {
                'filename': file_path.name,
            }

            documents.append(content)
            metadata_list.append(metadata)
            stats['processed'] += 1

            if len(documents) >= batch_size:
                rag_system.add_documents(documents, metadata_list)
                documents = []
                metadata_list = []

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    if documents:
        rag_system.add_documents(documents, metadata_list)

    return stats


if __name__ == "__main__":
    # Example usage:
    # python src/rag/add_to_db.py ./data/rag_data/rag_files/ rag_collection en
    parser = argparse.ArgumentParser(description='Process text files for RAG system')
    parser.add_argument('path', help='Path to file or directory to process')
    parser.add_argument('collection', help='Name of the ChromaDB collection')
    parser.add_argument('language', help='Language of the text files; en or pl')
    parser.add_argument('--db-dir', type=str, default='./chroma_db',
                        help='Directory for ChromaDB storage')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of files to process before persisting')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                        help='Do not recursively process subdirectories')

    args = parser.parse_args()

    rag = RAGSystem(
        collection_name=args.collection,
        language=args.language,
        persist_directory=args.db_dir,
    )

    # Process files
    path = Path(args.path)
    try:
        stats = process_files(
            rag_system=rag,
            path=path,
            recursive=args.recursive,
            batch_size=args.batch_size
        )
        print(f"\nProcessing complete. Statistics:")
        print(f"Total files: {stats['total']}")
        print(f"Successfully processed: {stats['processed']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
