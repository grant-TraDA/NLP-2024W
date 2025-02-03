import os.path

import fitz
import re
import nltk
from nltk.tokenize import sent_tokenize
import yaml
import glob
from tqdm import tqdm


# Download the required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def pdf_to_sentences(pdf_path, output_path, pages=None):
    """
    Convert specified pages of a PDF file to a text file with one sentence per line.

    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str): Path where the output text file will be saved
        pages (list, optional): List of page numbers to process (0-based). If None, process all pages.

    Returns:
        int: Number of sentences extracted
    """
    # Create PDF reader object
    pdf_document = fitz.open(pdf_path)

    # Get total number of pages
    total_pages = len(pdf_document)

    # If no pages specified, process all pages
    if pages is None:
        pages = range(total_pages)

    # Validate page numbers
    pages = [p for p in pages if 0 <= p < total_pages]

    # Extract text from specified pages
    text = ""
    for page_num in pages:
        text += pdf_document.load_page(page_num).get_text()

    # Clean up text
    # Remove commands
    text = re.sub(r'/[^\s]*', '', text)
    # Remove hyphenation at end of lines
    text = re.sub(r'\s*-\s*\n', '', text)
    text = re.sub(r'(?<=\w)-(?=\w)', '', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    # Split into sentences using NLTK
    sentences = sent_tokenize(text)

    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if not re.match(r'^[\s\d\W]+$', s)]

    # Write sentences to file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for sentence in sentences:
            out_file.write(sentence + '\n')

    return len(sentences)


# Example usage
if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    pdfs = config['pdfs']
    pdf_dir = config['pdf_dir']
    output_dir = config['output_dir']

    for path in tqdm(glob.glob(os.path.join(pdf_dir, '*.pdf'))):
        num_sentences = pdf_to_sentences(
            pdf_path=path,
            output_path=os.path.join(output_dir, os.path.basename(path).replace('.pdf', '.txt')),
        )

    """
    for pdf in pdfs:
        num_sentences = pdf_to_sentences(
            pdf_path=os.path.join(pdf_dir, pdf["name"]),
            output_path=os.path.join(output_dir, pdf["name"].replace('.pdf', '.txt')),
            pages=pdf["pages"]
        )
        print(f"Extracted {num_sentences} sentences")
    """


