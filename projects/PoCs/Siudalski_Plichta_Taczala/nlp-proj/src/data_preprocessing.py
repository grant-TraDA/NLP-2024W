import os
import re
import pandas as pd
import chardet
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pytesseract
from pdf2image import convert_from_path

# Initialize NLTK tools
from nltk import download
download('punkt')
download('stopwords')
download('wordnet')
lemmatizer = WordNetLemmatizer()

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Function to extract and clean text from .htm files
def extract_and_clean_text(htm_content):
    soup = BeautifulSoup(htm_content, "html.parser")
    raw_text = soup.get_text()  # Extract text from HTML
    raw_text = re.sub(r'\s+', ' ', raw_text)  # Remove extra whitespace
    raw_text = re.sub(r'^[^\w\s]{2,}$', '', raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', raw_text)  # Phone numbers
    raw_text = re.sub(r'\bP\.?O\.?\s?Box\b.*?\d+', '', raw_text, flags=re.IGNORECASE)  # PO Box
    raw_text = re.sub(r'[^\w\s,.]', '', raw_text)  # Retain alphanumeric characters, spaces, commas, and periods
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()  # Final cleanup
    return raw_text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return tokens

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for img in images[1:]:  # Skip the first page
            text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Failed to process OCR for {pdf_path}. Error: {e}")

    return text

base_directory = "datasets/Baza umów/"
processed_data = []

for state_folder in os.listdir(base_directory):
    state_path = os.path.join(base_directory, state_folder)

    if os.path.isdir(state_path):  # Check if it's a folder
        for file in os.listdir(state_path):

            file_path = os.path.join(state_path, file)
            if file.endswith(".htm"):
                continue
                #for now focus on pdfs
                
                # Process HTML files
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding']

                with open(file_path, "r", encoding=encoding, errors="replace") as file:
                    htm_content = file.read()
                    cleaned_text = extract_and_clean_text(htm_content)
                    tokens = preprocess_text(cleaned_text)

                    # Store processed data
                    processed_data.append({
                        "state": state_folder,
                        "file_name": file,
                        "cleaned_text": cleaned_text,
                        "tokens": tokens,
                    })

            elif file.endswith(".pdf"):

                # Process PDF files
                extracted_text = extract_text_from_pdf(file_path)
                cleaned_text = extract_and_clean_text(extracted_text)
                tokens = preprocess_text(cleaned_text)

                # Store processed data
                processed_data.append({
                    "state": state_folder,
                    "file_name": file,
                    "cleaned_text": cleaned_text,
                    "tokens": tokens,
                })
                print(f"Processed: {file_path}")

# Save results to a CSV file
output_dir = "outputs/preprocessed_data_csv"
os.makedirs(output_dir, exist_ok=True)
df = pd.DataFrame(processed_data)
df.to_csv(os.path.join(output_dir, "preprocessed_agreements_all_states.csv"), index=False)
