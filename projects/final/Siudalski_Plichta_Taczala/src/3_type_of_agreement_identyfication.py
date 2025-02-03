import pandas as pd
from transformers import pipeline

# Load the CSV file
input_file = "outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv"  # Replace with your file path
output_file = "outputs/tasks/3_type_of_agreement_identification_results.csv"
data = pd.read_csv(input_file)

# Initialize the zero-shot classification pipeline
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define potential agreement types
agreement_types = [
    "Memorandum of Understanding",
    "Sister Cities Agreement",
    "Trade Agreement",
    "Cultural Exchange Agreement",
    "Technology Transfer Agreement",
    "Educational Cooperation Agreement",
    "Bilateral Cooperation Agreement",
    "Partnership Agreement",
    "Investment Agreement",
]
import pandas as pd
from gpt4all import GPT4All
from tqdm import tqdm
import os
import csv


def extract_agreement_types(doc_text, model, chunk_size=4096, overlap=100):

    text_chunks = [
        doc_text[i : i + chunk_size]
        for i in range(0, len(doc_text), chunk_size - overlap)
    ]

    for idx, text_chunk in enumerate(text_chunks):
        with model.chat_session() as chat:
            prompt = (
                "You are an AI assistant trained to classify text from international agreements into predefined agreement types"
                "Your task is to analyze a provided document chunk and identify the type of agreement from the list: {agreement_types}\n\n"
                "Instructions:\n"
                "1. Read the document chunk carefully and classify it into one option from the list which is explicitly relevant.\n"
                "2. If no type is relevant then output 'Unknown'.\n"
                "\n"
                "Document Chunk:\n"
                f"{text_chunk}\n\n"
                "Output:\n"
                "Return a Python string with the name of the relevant agreement type. If agreement type is not on the list then return a string 'Unknown'."
            )
            response = chat.generate(prompt, max_tokens=1024)

            try:
                list_start_index = response.find("[")
                list_end_index = response.rfind("]")
                areas = response[list_start_index : list_end_index + 1]

                if areas:
                    return areas
                else:
                    continue
            except:
                return None
    return None


def main():
    print("Loading data...")
    file_path = "outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv"
    df = pd.read_csv(file_path).sort_values(by=["state", "file_name"]).head(5)
    df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)
    df["file_number"] = df.groupby("state").cumcount() + 1

    print("Initializing models...")
    llm_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="cpu")

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", newline="") as csvfile:
        fieldnames = ["state", "file_number", "file_name", "type_of_agreement"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0

        # Skip the first n rows
        # df = df.iloc[650:]

        for i, row in tqdm(df.iterrows(), total=len(df)):
            doc_text = row["cleaned_text"]
            agreement_types = extract_agreement_types(doc_text, llm_model)
            writer.writerow(
                {
                    "state": row["state"],
                    "file_number": row["file_number"],
                    "file_name": row["file_name"],
                    "type_of_agreement": agreement_types,
                }
            )

            count += 1
            if count % 50 == 0:
                csvfile.flush()

    print("Task completed.")


if __name__ == "__main__":
    main()
