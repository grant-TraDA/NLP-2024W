import pandas as pd

# Load the CSV file
file_path = 'outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv'
data = pd.read_csv(file_path)
f = data["tokens"]
data["number_of_words"] = data["tokens"].apply(lambda x: len(list(x)))







# # Print results
# print(f"Total Agreements: {total_agreements}")
# print(f"Sister Cities International Agreements: {sister_cities_agreements}")
# print(f"Percentage: {percentage_sister_cities:.2f}%")

# Save the results
output_path = 'outputs/tasks/number_of_words_results.csv'
data[["file_name","number_of_words"]].to_csv(output_path, index=False)


print("Classification results saved to:", output_path)
