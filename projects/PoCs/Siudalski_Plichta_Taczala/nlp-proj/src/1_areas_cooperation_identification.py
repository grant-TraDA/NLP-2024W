import pandas as pd
from gpt4all import GPT4All
from tqdm import tqdm
import re
from keybert import KeyBERT

def extract_keywords(doc_text, kw_model, top_n=15):
    """Extract relevant keywords and entities from the text."""
    keywords = kw_model.extract_keywords(doc_text, keyphrase_ngram_range=(2, 4), top_n=top_n)
    return [phrase for phrase, _ in keywords]

def identify_coop_areas(keywords, llm_model, n_predict=512):
    """Use LLM to identify actual cooperation areas from keywords."""
    with llm_model.chat_session() as chat:
        prompt = (
            "Based on the provided list of keywords and phrases from an international agreement, "
            "extract only the most relevant areas of cooperation. "
            "The areas must be short, specific, and relevant. "
            "Return only a concise list of cooperation areas, like:\n"
            "['Clean Energy', 'Environmental Protection', 'Research Collaboration', 'Funding Allocation'].\n"
            "Keywords and phrases:\n"
            f"{keywords}\n"
            "If no cooperation areas can be identified, return an empty list []."
        )

        output = chat.generate(prompt, n_predict=n_predict).strip()

        cooperation_areas = re.findall(r'\[(.*?)\]', output, re.DOTALL)
        
        if cooperation_areas:
            areas = re.findall(r'["\'](.*?)["\']', cooperation_areas[0])
            return list(set(area.strip() for area in areas))
        
        return []

def extract_coop_areas(doc_text, llm_model, kw_model):
    keywords = extract_keywords(doc_text, kw_model, top_n=20)
    cooperation_areas = identify_coop_areas(keywords, llm_model)

    if not cooperation_areas:
        print(f"No cooperation areas found for text: {doc_text[:100]}...")

    return cooperation_areas


def main():
    print("Loading data...")
    file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
    df = pd.read_csv(file_path)
    df = df.sample(n=50)
    df = df.reset_index(drop=True)
    df['cooperation_areas'] = None

    print("Initializing models...")
    llm_model = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf", device="cuda:NVIDIA GeForce RTX 3050 Ti Laptop GPU")
    kw_model = KeyBERT()

    print("Extracting cooperation areas...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        doc_text = row['cleaned_text']
        cooperation_areas = extract_coop_areas(doc_text, llm_model, kw_model)

        df.at[i, 'cooperation_areas'] = cooperation_areas

    print("Saving results...")
    output_path = 'outputs/tasks/identified_cooperation_areas.csv'
    df[['file_name', 'cooperation_areas']].to_csv(output_path, index=False)

    print("Task completed.")

if __name__ == "__main__":
    main()