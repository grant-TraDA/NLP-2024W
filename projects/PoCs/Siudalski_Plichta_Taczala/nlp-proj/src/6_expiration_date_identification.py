import pandas as pd
import spacy
from gpt4all import GPT4All
from tqdm import tqdm
import re

def extract_date_with_spacy(doc_text, model, spacy_nlp, chunk_size=1500, overlap=500):
    """Extract expiration date using spaCy and LLM."""

    text_chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size - overlap)]
    
    for idx, chunk in enumerate(text_chunks):
        doc = spacy_nlp(chunk)
        date_mentions = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        
        if date_mentions:
            with model.chat_session() as chat:
                prompt = (
                    "You are extracting the expiration date of an international agreement. "
                    "The expiration date refers to the specific date until which the agreement remains in effect. "
                    "This could be explicitly stated as a calendar date or described in relative terms, such as "
                    "'valid for five years from the signing date' or 'expires in December 2025.'\n\n"
                    "Extract the exact expiration date or date-like reference if present. "
                    "Return only the relevant information in the format:\n"
                    "**Effective until: {date}**\n"
                    "If no expiration date is clearly specified, return:\n"
                    "**Effective until: Not specified**\n\n"
                    "Text to analyze:\n"
                    f"{chunk}\n\n"
                    "Possible Dates Found: {date_mentions}\n"
                )

                output = chat.generate(prompt, n_predict=512).strip()
                
                if "Not specified" not in output:
                    return output
    
    return "**Effective until: Not specified**"

def main():
    print("Loading data...")
    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path)
    df = df.sample(n=50)
    df = df.reset_index(drop=True)
    df['valid_date'] = None

    print("Initializing models...")
    spacy_nlp = spacy.load("en_core_web_sm")
    model = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf", device="cuda:NVIDIA GeForce RTX 3050 Ti Laptop GPU")

    print("Extracting expiration dates...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        date_str = extract_date_with_spacy(row['cleaned_text'], model, spacy_nlp)
        date = re.search(r"Effective until: (.*?)\*\*?", date_str)

        df.at[i, 'valid_date'] = date.group(1) if date else date_str

    print("Saving output...")
    output_path = 'outputs/tasks/identified_expiration_dates.csv'
    df[['file_name', 'valid_date']].to_csv(output_path, index=False)

    print("Process completed.")
if __name__ == "__main__":
    main()