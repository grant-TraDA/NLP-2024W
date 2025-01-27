import pandas as pd
from gpt4all import GPT4All
from tqdm import tqdm
import os
import csv

def extract_coop_areas(doc_text, model, chunk_size=4096, overlap=100):
    """
    Extracts cooperation areas from a given document text.
    """
    text_chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size - overlap)]
    
    for idx, text_chunk in enumerate(text_chunks):
        with model.chat_session() as chat:
            prompt = (
                "You are an AI assistant trained to classify text from international agreements into predefined cooperation areas. "
                "Your task is to analyze a provided document chunk and identify which of the following predefined areas it relates to:\n"
                "['Economic Development and Trade', 'Education and Academic Exchange', 'Cultural Exchange', 'Tourism Promotion', "
                "'Environmental Protection and Sustainability', 'Infrastructure Development and Urban Planning', "
                "'Public Health and Social Services', 'Technology and Innovation', 'Disaster Preparedness and Emergency Management', "
                "'Good Governance and Administrative Cooperation', 'Agriculture'].\n\n"
                "Instructions:\n"
                "1. Read the document chunk carefully and classify it into one or more of the predefined areas, but only include areas that are explicitly relevant.\n"
                "2. If a text refers to a specific project, activity, or concept, ensure it is classified into the correct area(s).\n"
                "3. For each selected area, provide a short description of why it was chosen, adding this in parentheses next to the area name. "
                "For example: ['Economic Development and Trade (Promoting trade and investment opportunities)', "
                "'Education and Academic Exchange (Establishing student and faculty exchange programs)', "
                "'Environmental Protection and Sustainability (Sharing expertise in waste management and pollution control)'].\n"
                "4. Avoid over-classifying. Only choose areas that clearly align with the content of the text.\n"
                "5. If no areas apply, return an empty list [] without explanation.\n\n"
                "Document Chunk:\n"
                f"{text_chunk}\n\n"
                "Output:\n"
                "Return a Python list of the relevant areas with descriptions in parentheses as explained. If no areas apply, return an empty list [] without explanation."
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
    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path).sort_values(by=['state', 'file_name'])
    df = df.dropna(subset=['cleaned_text']).reset_index(drop=True)
    df['file_number'] = df.groupby('state').cumcount() + 1

    print("Initializing models...")
    llm_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="cuda:Quadro P5000")
    
    output_file = 'outputs/tasks/identified_cooperation_areas.csv'
    
    print("Extracting cooperation areas...")

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as csvfile:
        fieldnames = ['state', 'file_number', 'file_name', 'areas']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0

        # Skip the first n rows
        # df = df.iloc[650:]
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            doc_text = row['cleaned_text']
            areas = extract_coop_areas(doc_text, llm_model)
            writer.writerow({'state': row['state'], 'file_number': row['file_number'], 'file_name': row['file_name'], 'areas': areas})

            count += 1
            if count % 50 == 0:
                csvfile.flush()


    print("Task completed.")

if __name__ == "__main__":
    main()