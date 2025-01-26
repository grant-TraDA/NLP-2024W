import pandas as pd
from gpt4all import GPT4All
from tqdm import tqdm
import ast
import csv
import os

def extract_parties(doc_text, model, chunk_size=1500, overlap=500):
    """
    Extracts the contracting parties from a given document text.
    """
    text_chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size - overlap)]
    
    for idx, text_chunk in enumerate(text_chunks):
        with model.chat_session() as chat:
            prompt = (
                "You are an AI assistant trained to assist law experts in extracting contracting parties from international agreements. "
                "Your task is to extract the full names of the contracting parties, ensuring compound names are treated as single entities. "
                "Pay attention to political divisions, institutions, and geographic qualifiers (e.g., 'Jiangsu Province, Republic of China'). "
                "Recognize that institutions (e.g., 'Bank of England') should not be confused with geographic or sovereign entities (e.g., 'United Kingdom'). "
                "For entities from outside the United States, add the name of the country they are from after a comma. "
                "Analyze phrases like 'This agreement is between,' 'entered into by,' or 'signed by' to identify the involved parties. "
                "Always treat each party as a distinct and complete entity. "
                "Output strictly as a Python list of party names, e.g., ['Party 1', 'Party 2', 'Party 3']. "
                "If no relevant details are found, return strictly an empty list []. "
                "Example Input: 'This agreement is between the State of Alabama, United States and the Province of Ninh Thuan.' "
                "Example Output: ['State of Alabama, United States', 'Province of Ninh Thuan, Vietnam']. "
                "Example Input: 'This contract is signed by the Bank of England and the European Investment Bank.' "
                "Example Output: ['Bank of England, United Kingdom', 'European Investment Bank, European Union']. "
                "Example Input: 'This agreement involves Free and Sovereign State of Nuevo León and the State of California, United States.' "
                "Example Output: ['State of Nuevo León, Mexico', 'State of California, United States']. "
                "Under no circumstances should you output any other format than the one specified above. "
                f"Agreement Chunk: {text_chunk} "
            )       

            response = chat.generate(prompt, max_tokens=1024)

            try:
                list_start_index = response.find("[")
                list_end_index = response.rfind("]")
                parties = ast.literal_eval(response[list_start_index : list_end_index + 1])
                print(parties)

                if isinstance(parties, list) and parties:
                    return parties
                elif isinstance(parties, list) and not parties:
                    continue
                else:
                    return None
            except:
                print(response)
                return None


def main():
    print("Loading data...")

    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path).sort_values(by=['state', 'file_name'])
    df = df.dropna(subset=['cleaned_text']).reset_index(drop=True)
    df['file_number'] = df.groupby('state').cumcount() + 1
    df['parties'] = None

    print("Initializing models...")

    llm_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="cuda:Quadro P5000")

    output_file = 'outputs/tasks/identified_parties.csv'
    
    print("Extracting organization entities...")

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as csvfile:
        fieldnames = ['state', 'file_number', 'file_name', 'parties']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0

        # Skip the first n rows
        # df = df.iloc[650:]
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            doc_text = row['cleaned_text']
            parties = extract_parties(doc_text, llm_model)
            writer.writerow({'state': row['state'], 'file_number': row['file_number'], 'file_name': row['file_name'], 'parties': parties})

            count += 1
            if count % 50 == 0:
                csvfile.flush()
            

    print("Task completed.")

if __name__ == "__main__":
    main()