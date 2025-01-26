import pandas as pd

# Load the CSV file
file_path = "outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv"
data = pd.read_csv(file_path)


def count_words(text):
    return len(str(text).split())


# Add a new column to the DataFrame with word counts
data["number_of_words"] = data["cleaned_text"].apply(count_words)
# Save the results
output_path = "outputs/tasks/7_number_of_words_results.csv"
data["file_number"] = data.groupby("state").cumcount() + 1
data[["state", "file_number", "file_name", "number_of_words"]].to_csv(
    output_path, index=False
)


print("Classification results saved to:", output_path)
