import pandas as pd
import re
from collections import Counter
from transformers import pipeline

file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
data = pd.read_csv(file_path)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_agreement_types(text):
    matches = re.findall(
        r'\b(Memorandum of Understanding|Confidentiality Agreement|Trade Agreement|Cooperation Agreement|Agreement|MoU|Understanding|Protocol|Treaty|Accord|Partnership|Sister Cities Agreement)\b',
        str(text),
        re.IGNORECASE
    )
    return matches

data['agreement_type_matches'] = data['cleaned_text'].apply(lambda x: extract_agreement_types(x))

all_matches = [item.lower() for sublist in data['agreement_type_matches'] for item in sublist]
label_counts = Counter(all_matches)

labels = list(label_counts.keys())

def classify_agreement(tokens, threshold=0.3):
    text = " ".join(eval(tokens))
    result = classifier(text, labels)
    
    if result['scores'][0] >= threshold:
        return result['labels'][0]
    else:
        return "Unknown"

data['Agreement Type'] = data['tokens'].apply(classify_agreement)

def clean_file_name(file_name):
    pattern = r".*/([^/]+?\.\w{3}).*"
    match = re.search(pattern, file_name)
    if match:
        cleaned_string = match.group(1).split('.')[0]  # Get the base file name without extension
        return cleaned_string
    return file_name

data['file_name'] = data['file_name'].apply(clean_file_name)

# Debug: Print cleaned file names
print("Cleaned file_name values:")
print(data['file_name'].head(10))

output_path = 'outputs/tasks/type_of_agreement_identyfication_results.csv'
data[['file_name', 'Agreement Type']].to_csv(output_path, index=False)

print("Agreement types classified and saved to:", output_path)
