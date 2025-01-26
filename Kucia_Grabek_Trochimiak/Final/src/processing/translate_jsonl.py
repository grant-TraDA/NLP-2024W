import json
import time
import jsonlines
from openai import OpenAI
from typing import Dict, List
import os
from tqdm import tqdm


def translate_text(client: OpenAI, text: str) -> str:
    """
    Translate text from Polish to English using ChatGPT.
    Includes retry logic and rate limiting.
    """
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a translator. Translate the following Polish text to English. Maintain the same tone and style. Only return the translation, nothing else."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to translate after {max_retries} attempts: {str(e)}")
                return ""
            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff


def process_jsonl(input_file: str, output_file: str, api_key: str):
    """
    Process JSONL file, translating Polish text to English.
    Skip entries with empty outputs.
    """
    client = OpenAI(api_key=api_key)
    translated_data = []

    # Read and count lines first
    with jsonlines.open(input_file) as reader:
        total_lines = sum(1 for _ in reader)

    # Process the file with progress bar
    with jsonlines.open(input_file) as reader:
        for item in tqdm(reader, total=total_lines, desc="Translating"):
            # Skip entries with empty output
            if not item.get('output'):
                continue

            try:
                # Translate instruction and output
                translated_item = {
                    'instruction': translate_text(client, item['instruction']),
                    'input': item.get('input', ''),  # Keep empty input if it exists
                    'output': translate_text(client, item['output'])
                }

                # Only keep items where both translations succeeded
                if translated_item['instruction'] and translated_item['output']:
                    translated_data.append(translated_item)

                # Add a small delay to respect rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing entry: {str(e)}")
                continue

    # Write translated data
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(translated_data)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Translate Polish JSONL to English using ChatGPT')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env variable)')

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable")

    process_jsonl(args.input_file, args.output_file, api_key)
    print(f"Translation completed. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
