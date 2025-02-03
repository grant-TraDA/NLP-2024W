import os.path
import argparse

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


def pdf_to_sentences(pdf_path: str, output_path: str) -> int:
    """
    Convert a PDF file to a text file with one sentence per line.

    :param pdf_path: Path to the input PDF file
    :param output_path: Path where the output text file will be saved
    :return: Number of sentences extracted
    """

    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)

    text = ""
    for page_num in range(total_pages):
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

    with open(output_path, 'w', encoding='utf-8') as out_file:
        for sentence in sentences:
            out_file.write(sentence + '\n')

    return len(sentences)


if __name__ == "__main__":
    # Example usage:
    # python src/processing/pdf_to_txt.py --pdf_dir data/pdf --output_dir data/txt
    parser = argparse.ArgumentParser(description='Convert PDFs to sentences')
    parser.add_argument('--pdf_dir', type=str, required=True,
                        help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output text files')

    args = parser.parse_args()

    for path in tqdm(glob.glob(os.path.join(args.pdf_dir, '*.pdf'))):
        num_sentences = pdf_to_sentences(
            pdf_path=path,
            output_path=os.path.join(args.output_dir, os.path.basename(path).replace('.pdf', '.txt')),
        )
