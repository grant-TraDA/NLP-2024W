import pandas as pd
from tqdm import tqdm
import os
import csv
from gpt4all import GPT4All
import nltk
from nltk.tokenize import sent_tokenize
from flair.data import Sentence
from flair.models import SequenceTagger



def extract_dates(doc_text, model, tagger, chunk_size=1500, overlap=200):
    """
    Extracts the signing and validity dates from a given document text.
    """

    text_chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size - overlap)]
    date_sentences = []
    
    for idx, text_chunk in enumerate(text_chunks):
        sentences = sent_tokenize(text_chunk)
        entities = []

        for sent in sentences:
            sentence = Sentence(sent)
            tagger.predict(sentence)
            date_mentions = [entity.text for entity in sentence.get_spans('ner') if entity.tag in ['DATE']]
            
            if date_mentions:
                date_sentences.append(sent)
                
    if not date_sentences:
        return str({"signing_date": "Not specified", "validity_date": "Not specified"})
                   
    with model.chat_session() as chat:
        prompt = (
            "You are an AI assistant trained to extract critical information about international agreements. "
            "Your task is to identify the signing date and the expiration or validity date of the agreement. "
            "The signing date refers to the date when the agreement was signed. The expiration or validity date refers to the specific date "
            "or duration until which the agreement remains in effect.\n\n"
            "Details to extract:\n"
            "- Signing Date: Look for phrases such as 'signed on [date]' or similar references. If no signing date is mentioned, return 'Not specified'.\n"
            "- Validity Date: Identify the specific date or relative duration (e.g., 'valid for five years from the signing date' or 'expires in December 2025'). If no validity date is mentioned, return 'Not specified'.\n\n"
            "Return the extracted information strictly in the following JSON format:\n"
            "{\n"
            "  \"signing_date\": \"[DD-MM-YYYY]\" or \"Not specified\",\n"
            "  \"validity_date\": \"[DD-MM-YYYY]\", \"[Relative term]\", or \"Not specified\"\n"
            "}\n\n"
            "Examples:\n"
            "1. Input: 'This agreement was signed on March 1, 2000 and is valid until March 1, 2005.'\n"
            "   Output: {\n"
            "     \"signing_date\": \"01-03-2000\",\n"
            "     \"validity_date\": \"01-03-2005\"\n"
            "   }\n\n"
            "2. Input: 'The agreement will expire five years after signing but does not specify the signing date.'\n"
            "   Output: {\n"
            "     \"signing_date\": \"Not specified\",\n"
            "     \"validity_date\": \"Valid for five years after signing\"\n"
            "   }\n\n"
            "3. Input: 'This agreement has no specific expiration date and was signed on January 15, 2010.'\n"
            "   Output: {\n"
            "     \"signing_date\": \"15-01-2010\",\n"
            "     \"validity_date\": \"Not specified\"\n"
            "   }\n\n"
            f"Possible date mentions: {date_sentences}"
        )
        
        response = chat.generate(prompt, n_predict=512).strip()
        
        output = str(response)
        ans_start_index = output.find("{")
        ans_end_index = output.rfind("}")
        final_dates = output[ans_start_index : ans_end_index + 1]
        if not final_dates:
            return str({"signing_date": "Not specified", "validity_date": "Not specified"})

        return final_dates
    
def main():
    print("Loading data...")

    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path).sort_values(by=['state', 'file_name'])
    df = df.dropna(subset=['cleaned_text']).reset_index(drop=True)
    df['file_number'] = df.groupby('state').cumcount() + 1

    print("Initializing models...")

    llm_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="cuda:Quadro P5000")
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    nltk.download('punkt')
    
    output_file = 'outputs/tasks/identified_dates.csv'
    
    print("Extracting signing and validity dates...")

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as csvfile:
        fieldnames = ['state', 'file_number', 'file_name', 'dates']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0

        # Skip the first n rows
        # df = df.iloc[600:]
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            doc_text = row['cleaned_text']
            date_str = extract_dates(doc_text, llm_model, tagger)
            writer.writerow({'state': row['state'], 'file_number': row['file_number'], 'file_name': row['file_name'], 'dates': date_str})

            count += 1
            if count % 50 == 0:
                csvfile.flush()
            

    print("Task completed.")

if __name__ == "__main__":
    main()