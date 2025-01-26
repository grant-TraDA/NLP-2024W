import json
import os
from typing import List, Dict
import jsonlines


def process_conversation(messages: List[Dict]) -> List[Dict]:
    """
    Process a list of messages from a conversation, applying feedback rules.
    Returns a list of QA pairs suitable for training.
    """
    qa_pairs = []

    for message in messages:
        # Skip messages with null feedback
        if message.get('feedback') is None:
            continue

        user_input = message['user_input'].strip()

        # Handle feedback cases
        feedback = message['feedback']
        if feedback['type'] == 'thumbs_down':
            # Use the feedback comment as the correct response
            correct_response = feedback['comment'].strip()
        elif feedback['type'] == 'thumbs_up':
            # Use the assistant's response as the correct response
            correct_response = message['assistant_response'].strip()
        else:
            continue

        # Create training example in LLaMA format
        training_example = {
            "instruction": user_input,
            "input": "",  # LLaMA can handle empty input
            "output": correct_response
        }

        qa_pairs.append(training_example)

    return qa_pairs


def process_json_file(file_path: str) -> List[Dict]:
    """
    Process a single JSON file and return formatted QA pairs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'messages' not in data:
            print(f"Warning: Invalid format in file {file_path}")
            return []

        return process_conversation(data['messages'])

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []


def main(input_dir: str, output_file: str):
    """
    Process all JSON files in the input directory and write results to output file.
    """
    all_qa_pairs = []

    # Process each JSON file in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            qa_pairs = process_json_file(file_path)
            all_qa_pairs.extend(qa_pairs)

    # Write results to JSONL file
    with jsonlines.open(output_file, mode='w') as writer:
        for qa_pair in all_qa_pairs:
            writer.write(qa_pair)

    print(f"Processed {len(all_qa_pairs)} QA pairs from {input_dir}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert JSON conversations to LLaMA fine-tuning format')
    parser.add_argument('input_dir', help='Directory containing JSON files')
    parser.add_argument('output_file', help='Output JSONL file path')

    args = parser.parse_args()

    main(args.input_dir, args.output_file)
