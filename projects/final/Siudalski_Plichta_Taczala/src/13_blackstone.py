#To run following code you need to use 3.6 version of Python
#To install blackstone :
# 1. Clone github blackstone repository 
# 2. pip install spacy==2.1.9 (To ensure that this won't raise any errors)
# 3. pip install --editable .
# 4. pip install https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz

#To run the code python blackstone.py preprocessed_agreements_all_states-2.csv output_file.csv
import spacy
import pandas as pd
import argparse

def process_citations(input_file, output_file):
    nlp = spacy.load("en_blackstone_proto")
    
    df = pd.read_csv(input_file)
    
    citations_data = []
    
    for index, row in df.iterrows():
        text = row['cleaned_text']
        doc = nlp(text)
        
        # Get all citations for this text
        text_citations = [ent.text for ent in doc.ents if ent.label_ == "CITATION"]
        
        citations_data.append({
            'citations': ';'.join(text_citations) if text_citations else 'NA',
            'citation_count': len(text_citations) if text_citations else 0
        })
    
    citations_df = pd.DataFrame(citations_data)
    
    citations_df.to_csv(output_file, index=False)
    
    return len(df), sum(1 for d in citations_data if d['citations'] != 'NA'), sum(d['citation_count'] for d in citations_data)

def main():
    parser = argparse.ArgumentParser(description='Process text files to extract citations')
    parser.add_argument('input_file', help='Path to input CSV file containing cleaned_text column')
    parser.add_argument('output_file', help='Path where output CSV file should be saved')
    
    args = parser.parse_args()
    
    try:
        total_texts, texts_with_citations, total_citations = process_citations(
            args.input_file, 
            args.output_file
        )
        
        print(f"Successfully processed {total_texts} texts")
        print(f"Found citations in {texts_with_citations} texts")
        print(f"Total number of citations: {total_citations}")
        print(f"Results saved to {args.output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()