import pandas as pd
from gpt4all import GPT4All


def analyze_clauses_gpt4all(csv_file_path, model_name):
    # Load the CSV file into a DataFrame
    print("Loading CSV file...")
    df = pd.read_csv(csv_file_path).head(
        10
    )  # Load only the first 10 rows for demonstration purposes
    print("CSV file loaded successfully. Number of rows:", len(df))

    # Load the GPT4All model
    print("Loading GPT4All model...")
    model = GPT4All(model_name)
    print("Model loaded successfully.")

    results = []

    print("Processing agreements...")
    for idx, text in enumerate(df["cleaned_text"]):
        print(f"Processing row {idx + 1}...")

        # Query GPT4All model to classify the clause
        prompt = (
            f"Classify the following clause into one of three categories based on the strength and frequency of commitment expressed:\n"
            f"- Always: Indicates a mandatory or binding obligation that must always be fulfilled, such as clauses with 'shall' or 'must.'\n"
            f"- Often: Indicates a frequent but not absolute obligation or intent, such as clauses with 'promote' or 'encourage.'\n"
            f"- Rarely: Indicates an optional or discretionary action that is seldom required, such as clauses with 'may' or 'consider.'\n\n"
            f"Clause: {text}\n\n"
            f"Answer with one word only: Always, Often, or Rarely."
        )

        # Use the `generate` method for text generation
        response = model.generate(prompt)
        print(f"Model response: {response}")

        # Parse the response (this may vary depending on the model's output format)
        results.append(
            {
                "file_name": df.loc[idx, "file_name"],
                "clause": text,
                "classification": response.strip(),
            }
        )

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    print("Classification completed. Results DataFrame created:")
    print(results_df)

    return results_df


# Example usage
if __name__ == "__main__":
    csv_file_path = (
        "outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv"
    )
    model_name = "gpt4all-13b-snoozy-q4_0.gguf"  # Replace with the appropriate GPT4All model name

    print("Starting clause frequency analysis with GPT4All...")
    result_df = analyze_clauses_gpt4all(csv_file_path, model_name)

    # Save the result to a CSV file
    result_csv_path = "gpt4all_clause_analysis_results.csv"
    result_df.to_csv(result_csv_path, index=False)
    print(f"Clause frequency analysis completed. Results saved to '{result_csv_path}'.")
