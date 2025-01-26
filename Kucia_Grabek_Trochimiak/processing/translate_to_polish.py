import os
import re
import argparse
from openai import OpenAI
from typing import List, Set
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

MODEL_NAME = "gpt-4o-mini"


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text by dividing character count by 3."""
    return len(text) // 3


def chunk_text(lines: List[str], max_tokens: int) -> List[List[str]]:
    """Split lines into chunks that don't exceed the token limit."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)

        if current_tokens + line_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def translate_chunk(chunk: List[str], client: OpenAI, prompt: str) -> str:
    """Translate a chunk of text using OpenAI API."""
    text_to_translate = "\n".join(chunk)
    full_prompt = f"{prompt}\n\nText to translate:\n{text_to_translate}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


def get_files_to_process(input_dir: str, output_dir: str, min_chars: int = 3000) -> List[str]:
    """Get list of files that need processing, excluding already processed ones and small files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get set of already processed files
    processed_files = {f.name for f in Path(output_dir).glob('*')}

    files_to_process = []
    for file_path in Path(input_dir).glob('*'):
        if file_path.is_file() and file_path.name not in processed_files:
            try:
                # Check file size
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) >= min_chars:
                        files_to_process.append(str(file_path))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    return files_to_process


def translate_file(
        input_path: str,
        output_dir: str,
        prompt: str,
        api_key: str,
        max_tokens: int = 10000
) -> None:
    """Translate a single file."""
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    try:
        # Get output path
        output_path = os.path.join(output_dir, Path(input_path).name)

        # Read input file
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Split into chunks
        chunks = chunk_text(lines, max_tokens)

        # Translate each chunk and write to output file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for i, chunk in enumerate(chunks, 1):
                print(f"[PID {os.getpid()}] Translating {Path(input_path).name} - chunk {i} of {len(chunks)}...")

                translated_text = translate_chunk(chunk, client, prompt)
                if translated_text:
                    cleaned_text = re.sub(r'\n+', '\n', translated_text)
                    out_file.write(cleaned_text + '\n')
                else:
                    print(f"Failed to translate chunk {i} of {input_path}")

        print(f"Translation completed for {input_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_files(files: List[str], output_dir: str, prompt: str, api_key: str, num_processes: int = 15):
    """Process files in parallel using multiple processes."""
    # Create partial function with fixed arguments
    translate_func = partial(translate_file, output_dir=output_dir, prompt=prompt, api_key=api_key)

    # Create process pool and process files
    with mp.Pool(processes=num_processes) as pool:
        pool.map(translate_func, files)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Translate text files using OpenAI API')
    parser.add_argument('input_dir', help='Path to the input directory containing text files')
    parser.add_argument('--min-chars', type=int, default=3000,
                        help='Minimum number of characters for processing (default: 3000)')
    parser.add_argument('--processes', type=int, default=15, help='Number of parallel processes (default: 15)')

    args = parser.parse_args()

    # Hardcoded output directory
    OUTPUT_DIR = "../data/txt_translation_english"

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    # Translation prompt
    # prompt = """W wiadomości poniżej dostaniesz zdań tekstu w różnych językach. Każda linijka z założenia to jedno zdania.
    # Twoim zadaniem jest przetłumaczenie ich na język polski. W każdej linijce mogą zdarzyć błędy językowe, mogą być też
    # jakieś dodatkowe 'śmieci', których nienależy tłumaczyć tylko usunąć. Może być też tak, że cała linijka jest zbędna,
    # wtedy ją usuń. Ignoruj wszystkie przypiski od wydawców które nie dodają merytoryki. Twoją odpowiedzią powinien być
    # tylko tekst przetłumaczony i zredakowany. Każda linijka odpowiedzi to jedno przetłumaczone zdanie, nie dodawaj nic
    # innego. Zdania do przetłumaczenia są poniżej:\n\n"""

    prompt = "Translate the message below to english. Each line should be exactly one translated sentence, nothing more. The text to translate:"

    # Get list of files to process
    files_to_process = get_files_to_process(args.input_dir, OUTPUT_DIR, args.min_chars)

    if not files_to_process:
        print("No files to process!")
        exit(0)

    print(f"Found {len(files_to_process)} files to process")

    # Start processing
    try:
        process_files(files_to_process, OUTPUT_DIR, prompt, api_key, args.processes)
        print("All processing completed successfully!")
    except Exception as e:
        print(f"An error occurred during processing: {e}")