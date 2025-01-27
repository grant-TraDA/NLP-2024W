import requests
import time
import os
from tqdm import tqdm

HF_API_TOKEN = ""

HF_API_URL = ""



headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def query_hf_endpoint(prompt, max_new_tokens=200, temperature=0.5):
    """
    Query the Hugging Face Inference Endpoint with a given prompt.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for non-200 responses
        result = response.json()
        return result[0]["generated_text"]
    except Exception as e:
        print(f"Error querying the Hugging Face endpoint: {e}")
        raise

def ask_multiple_questions_sequentially(input_file, output_file, book_title):
    """
    Reads questions from a text file, queries the Hugging Face endpoint, and saves answers.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            questions = file.readlines()

        with open(output_file, "w", encoding="utf-8") as file:
            for question in tqdm(questions, desc=f"Processing {os.path.basename(input_file)}", unit="question"):
                question = question.strip()
                if not question:
                    continue

                prompt = f"Dostajesz pytanie o książce \"{book_title}\".\nPytanie: {question}\nProszę, napisz zwięzłą odpowiedź."
                try:
                    response = query_hf_endpoint(prompt)
                    stripped_answer = response.split("\n", 1)[-1].strip()
                    file.write(f"{stripped_answer}\n===\n")
                except Exception as e:
                    file.write("Error retrieving answer.\n===\n")
                    print(f"Error for question: '{question}'\n{e}")
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

def process_folder(input_folder, output_folder):
    """
    Processes multiple question files in a folder using the Hugging Face endpoint.
    """
    os.makedirs(output_folder, exist_ok=True)
    question_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for filename in tqdm(question_files, desc="Processing files", unit="file"):
        try:
            base_name = os.path.splitext(filename)[0]
            book_title = " ".join(base_name.split("_"))

            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{base_name}_answers.txt")

            if os.path.exists(output_file):
                print(f"Skipping {filename}, already processed.")
                continue

            print(f"Processing: {filename} for book: {book_title}")
            ask_multiple_questions_sequentially(input_file, output_file, book_title)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    try:
        input_folder = "../data_processing_scripts/q_a/questions"
        output_folder = "../data_processing_scripts/q_a/answers_hf"

        print("Starting processing of files using Hugging Face endpoint...")
        process_folder(input_folder, output_folder)

        print(f"All answers saved to '{output_folder}'.")
    except Exception as e:
        print(f"Critical error: {e}")
