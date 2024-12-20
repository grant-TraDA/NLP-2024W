import pandas as pd
import ast
import random

# Read the original CSV file
df = pd.read_csv('processed-data/labeled_files.csv')

# Convert string representations of lists to actual lists
df['matching_labels'] = df['matching_labels'].apply(ast.literal_eval)
df['non_matching_labels'] = df['non_matching_labels'].apply(ast.literal_eval)

# Create lists to store our results
markdown_contents = []
matching_labels = []
non_matching_labels = []
matching_statuses = []

# Process each row
for idx, row in df.iterrows():
    markdown_contents.append(row['markdown_content'])
    
    # 50% chance for matching or non-matching
    if random.random() < 0.5:
        # Choose matching label
        matching_labels.append(random.choice(row['matching_labels']))
        non_matching_labels.append('')  # Empty string for non-matching
        matching_statuses.append(True)
    else:
        # Choose non-matching label
        matching_labels.append('')  # Empty string for matching
        non_matching_labels.append(random.choice(row['non_matching_labels']))
        matching_statuses.append(False)

# Create new dataframe with the processed data
processed_df = pd.DataFrame({
    'markdown_content': markdown_contents,
    'matching_label': matching_labels,
    'non_matching_label': non_matching_labels,
    'matching_status': matching_statuses
})

# Save the processed dataframe to a new CSV file
processed_df.to_csv('processed-data/ready_dataset.csv', index=False)

# Print first few rows to verify the result
print("First few rows of the processed dataset:")
print(processed_df.head())

# Print shape of the dataset
print("\nDataset shape:", processed_df.shape)

# Print distribution of matching status
print("\nDistribution of matching status:")
print(processed_df['matching_status'].value_counts(normalize=True))