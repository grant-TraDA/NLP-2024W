from gpt4all import GPT4All
import pandas as pd
import sys
import os
from tqdm import tqdm  

def create_detection_prompt(agreement_text):
    """Create a structured prompt for initial detection"""
    return f"""Analyze the following agreement text and determine if it includes specific provisions
    for evaluating its implementation. Focus on elements like timelines of meetings or events,
    review periods, or assessment criteria. If they're exisitng answer YES and list all the actions mentioned in the document. If
    there aren't any write NO.
    Agreement text:
    {agreement_text}
    """

def process_agreements(input_file, output_file, model_name="Phi-3-mini-4k-instruct.Q4_0.gguf", num_samples=20):
    try:
        # Load the model
        print(f"Loading model: {model_name}")
        model = GPT4All(model_name)
        
        # Load the input data
        print(f"Loading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        if len(df) == 0:
            raise ValueError("Input file is empty")
            
        # Adjust num_samples if it's larger than the dataset
        num_samples = min(num_samples, len(df))
        print(f"Processing {num_samples} samples")
        
        # Initialize evaluation DataFrame
        evaluation = []
        evaluation_df = pd.DataFrame(evaluation, columns=['evaluation_yes_no'])
        
        # Process agreements with progress bar
        for i in tqdm(range(num_samples), desc="Processing agreements"):
            with model.chat_session():
                detection_prompt = create_detection_prompt(df["cleaned_text"][i])
                response = model.generate(detection_prompt, max_tokens=1024)
                evaluation_df.loc[len(evaluation_df)] = [response]
        
        # Save results
        print(f"Saving results to: {output_file}")
        evaluation_df.to_csv(output_file, index=False)
        print("Processing completed successfully")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def main():
    """Main function to handle command line arguments and run the script"""
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.csv output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    # Validate output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Process the agreements
    process_agreements(input_file, output_file)

if __name__ == "__main__":
    main()