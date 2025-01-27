from typing import Optional, Dict
import argparse
import os
from pathlib import Path


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text using a basic tokenization approach
    similar to GPT models. This is a simplified implementation and may not exactly
    match GPT-4's tokenization, but provides a reasonable approximation.

    Args:
        text (str): Input text to tokenize

    Returns:
        int: Approximate number of tokens
    """
    return int(len(text) / 3)


def process_file(file_path: str) -> Optional[int]:
    """
    Read a text file and count its tokens.

    Args:
        file_path (str): Path to the text file

    Returns:
        Optional[int]: Number of tokens if successful, None if file reading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return count_tokens(text)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None


def process_path(path: str) -> Dict[str, int]:
    """
    Process a file or directory path and count tokens.

    Args:
        path (str): Path to file or directory

    Returns:
        Dict[str, int]: Dictionary mapping file paths to token counts
    """
    results = {}
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() == '.txt':
            token_count = process_file(str(path))
            if token_count is not None:
                results[str(path)] = token_count

    elif path.is_dir():
        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() == '.txt':
                    token_count = process_file(str(file_path))
                    if token_count is not None:
                        results[str(file_path)] = token_count

    return results


def main():
    parser = argparse.ArgumentParser(description='Count tokens in text files')
    parser.add_argument('path', help='Path to file or directory')

    args = parser.parse_args()

    # Process the path
    results = process_path(args.path)

    if not results:
        print("No text files found or processed.")
        return

    # Print results for each file
    for file_path, count in sorted(results.items()):
        print(f"\nFile: {file_path}")
        print(f"Total tokens: {count:,}")

    # Print total if processing multiple files
    if len(results) > 1:
        total_tokens = sum(results.values())
        print(f"\nTotal files processed: {len(results)}")
        print(f"Total tokens across all files: {total_tokens:,}")


if __name__ == "__main__":
    main()
