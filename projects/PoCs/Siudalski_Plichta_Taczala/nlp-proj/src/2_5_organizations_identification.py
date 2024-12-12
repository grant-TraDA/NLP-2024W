import pandas as pd
import spacy
from gpt4all import GPT4All
from tqdm import tqdm
import re
import json
from fuzzywuzzy import fuzz

def deduplicate_entities(entities, threshold=85):
    deduplicated = []
    for entity in entities:
        if not any(fuzz.ratio(entity.lower(), e.lower()) > threshold for e in deduplicated):
            deduplicated.append(entity)
    return deduplicated

def clean_and_parse_json(response_str):
    cleaned_response = re.sub(r'//.*?(\n|$)', '', response_str)
    cleaned_response = re.sub(r'[\n\r]', ' ', cleaned_response)
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
    
    json_match = re.search(r'(\[.*?\])', cleaned_response, re.DOTALL)
    
    if json_match:
        extracted_json_str = json_match.group(1).strip()

        extracted_json_str = re.sub(r'[^,\[\]\{\}":\w\s]', '', extracted_json_str)
        
        extracted_json_str = re.sub(r',\s*(\])', r'\1', extracted_json_str)
        extracted_json_str = re.sub(r'(?<=\})\s*(?=\{)', ',', extracted_json_str)
        
        try:
            extracted_json = json.loads(extracted_json_str)
            return extracted_json
        except json.JSONDecodeError as e:
            print("Original Response:", response_str)
            print("Cleaned Response:", extracted_json_str)
            print(f"Failed to parse JSON: {e}")
            return None
    else:
        print("No JSON found.")
        return None
    
def extract_org_entities(doc_text, spacy_nlp, model):
    doc_spacy = spacy_nlp(doc_text)
    spacy_entities = set([ent.text for ent in doc_spacy.ents if ent.label_ == "ORG"])
    spacy_entities = deduplicate_entities(list(spacy_entities))
    
    if spacy_entities:
        with model.chat_session() as chat:
            prompt = (
                "You are a fact-checking assistant specializing in identifying real organizations and institutions. "
                "Given the following list of potential entities, some of which may be random text or irrelevant terms, "
                "return only verified organizations and international institutions as a valid JSON array. "
                "Your output should **strictly** follow this format, with no extra explanations, comments, or placeholder text: "
                '[{"name": "Full Organization Name", "abbreviation": "Abbreviation"}, '
                '{"name": "Organization Without Abbreviation", "abbreviation": null}]. '
                "If the abbreviation is not applicable, use 'null'.\n"
                f"Entities to check: {spacy_entities}"
            )
            
            output = chat.generate(prompt, max_tokens=1024)
            return output


def main():
    print("Loading data...")
    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path)
    df = df.sample(n=50)
    df = df.reset_index(drop=True)
    df['entities'] = None

    print("Initializing models...")
    spacy_nlp = spacy.load("en_core_web_sm")
    llm_model = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf", device="cuda:NVIDIA GeForce RTX 3050 Ti Laptop GPU")

    print("Extracting organization entities...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        doc_text = row['cleaned_text']
        entities = extract_org_entities(doc_text, spacy_nlp, llm_model)
        parsed_entities = clean_and_parse_json(entities)
        df.at[i, 'entities'] = parsed_entities
    
    print("Saving results...")
    output_path = 'outputs/tasks/identified_entities.csv'
    df[['file_name', 'entities']].to_csv(output_path, index=False)

    print("Task completed.")

if __name__ == "__main__":
    main()