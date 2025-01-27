import pandas as pd
from gpt4all import GPT4All
from tqdm import tqdm
import os
import csv
from fuzzywuzzy import fuzz
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.tokenize import sent_tokenize
import ast
import nltk
nltk.download('punkt')



def safe_literal_eval(value):
    """"
    Safely evaluate a string containing a Python literal or container.
    """
    try:
        if pd.isna(value):
            return None
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None

    
def deduplicate_entities(extracted_entities, excluded_entities, threshold=95):
    """
    Deduplicates a list of extracted entities based on semantic similarity with a list of excluded entities.
    """
    deduplicated = []
    for entity in extracted_entities:
        normalized_entity = entity.strip().lower()
        for excl_entity in excluded_entities:
            normalized_excl_entity = excl_entity.strip().lower()
            if fuzz.token_set_ratio(normalized_entity, normalized_excl_entity) >= threshold:
                break
        else:
            deduplicated.append(entity)
    return deduplicated


def extract_org_entities(doc_text, tagger, model, excluded_parties):
    """
    Extracts organization entities from a given document text and filters out irrelevant or duplicate entities.
    """
    sentences = sent_tokenize(doc_text)
    entities = []

    for sent in sentences:
        sentence = Sentence(sent)
        tagger.predict(sentence)
        entities.extend([entity.text for entity in sentence.get_spans('ner') if entity.tag in ['ORG']])
    
    if entities:
        with model.chat_session() as chat:
            prompt = (
                "You are an AI assistant specializing in extracting and verifying valid organization and institution names from text. "
                "Your task is to review a list of potential organization names extracted using an automated method. "
                "These names may be broken, incomplete, invalid, or irrelevant. Your job is to:\n"
                "1. Verify the validity of each organization name and correct incomplete or broken names using your internal knowledge.\n"
                "2. Remove irrelevant, invalid, or generic terms that are not real organizations.\n"
                "3. Compare the extracted entities with the provided list of previously identified parties, and exclude any entities that refer to the same organization "
                "even if their names are not an exact match. Use semantic similarity and common abbreviations to identify matches "
                "(e.g., 'Government of the State of California' and 'Government of California' or 'UNESCO' and 'United Nations Educational, Scientific and Cultural Organization').\n"
                "4. Prioritize providing full and accurate names of organizations where possible. If a name is ambiguous or incomplete, use your knowledge to infer the correct full name.\n"
                "5. Return a final Python list of verified and corrected full organization names, ensuring no duplicates.\n\n"
                "Details to process:\n"
                f"Extracted Entities: {entities}\n"
                f"Previously Identified Parties to Ignore: {excluded_parties}\n\n"
                "Carefully review the extracted entities and ensure all names are valid, accurate, and complete. "
                "Exclude any entity that matches or overlaps with those in the 'Previously Identified Parties to Ignore' section, even if their names differ slightly. "
                "Use your internal knowledge and reasoning to identify and exclude duplicates. "
                "Return only the final Python list of verified organization names, with no additional text or explanation."
            )
            
            response = chat.generate(prompt, max_tokens=1024)

            if not response or "[" not in response or "]" not in response:
                print("Invalid or empty response from chat.generate")
                return []

            try:
                list_start_index = response.find("[")
                list_end_index = response.rfind("]")
                output = ast.literal_eval(response[list_start_index : list_end_index + 1])
            except (ValueError, SyntaxError) as e:
                print(f"Error evaluating response: {e}")
                return []

            if isinstance(output, list):
                return output
            else:
                print("Output is not a valid list")
                return []

    return []



def main():
    print("Loading data...")

    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path)
    
    df_parties = pd.read_csv('outputs/tasks/identified_parties.csv')
    df = df.merge(df_parties, on=['file_name', 'state'], how='left')
    
    df = df.dropna(subset=['cleaned_text']).reset_index(drop=True).sort_values(by=['state', 'file_name'])

    print("Initializing models...")

    llm_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="cuda:Quadro P5000")
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    output_file = 'outputs/tasks/identified_organizations.csv'
    
    print("Extracting organization entities...")

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as csvfile:
        fieldnames = ['state', 'file_number', 'file_name', 'organizations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0

        # Skip the first n rows
        # df = df.iloc[645:]
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            doc_text = row['cleaned_text']
            parties = safe_literal_eval(row['parties'])
            excluded_parties = [party.split(', ')[0] for party in parties] if parties else []
            extracted_entities = extract_org_entities(doc_text, tagger, llm_model, excluded_parties)
            entities = deduplicate_entities(extracted_entities, excluded_parties)
            writer.writerow({'state': row['state'], 'file_number': row['file_number'], 'file_name': row['file_name'], 'organizations': entities})

            count += 1
            if count % 50 == 0:
                csvfile.flush()
            

    print("Task completed.")

if __name__ == "__main__":
    main()