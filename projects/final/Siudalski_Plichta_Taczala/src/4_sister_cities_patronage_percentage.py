import pandas as pd
from transformers import pipeline

# Load the CSV file
file_path = "outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv"
data = pd.read_csv(file_path)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for classification
labels = ["YES", "NO"]

# Step 1: Classify agreements using zero-shot classification
# Determine if the following agreement is under the patronage of Sister Cities International


CONFIDENCE_THRESHOLD = 0.7


def classify_sister_cities(text):
    try:
        result = classifier(
            f"Determine if the following agreement is under the patronage of Sister Cities International: {text}",
            labels,
        )
        # Return the label with the highest confidence
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        # Apply confidence threshold
        if top_score >= CONFIDENCE_THRESHOLD:
            return top_label
        else:
            return False
    except Exception as e:
        return "Error"


# Apply classification to the cleaned_text column
data["Sister Cities Classification"] = data["cleaned_text"].apply(
    lambda x: classify_sister_cities(str(x))
)

# Step 2: Flag agreements under Sister Cities International patronage
data["Is Sister Cities International"] = data["Sister Cities Classification"].apply(
    lambda x: x == labels[0]
)

# Step 3: Calculate percentage
total_agreements = len(data)
sister_cities_agreements = data["Is Sister Cities International"].sum()
percentage_sister_cities = (sister_cities_agreements / total_agreements) * 100

# Print results
print(f"Total Agreements: {total_agreements}")
print(f"Sister Cities International Agreements: {sister_cities_agreements}")
print(f"Percentage: {percentage_sister_cities:.2f}%")

# Save the results
output_path = "outputs/tasks/4_sister_cities_classification_results.csv"
data["file_number"] = data.groupby("state").cumcount() + 1
data[["state", "file_name", "Is Sister Cities International"]].to_csv(
    output_path, index=False
)

print("Classification results saved to:", output_path)
