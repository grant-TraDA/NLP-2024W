import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('processed-data/processed-files.csv')

# Calculate lengths of markdown content
df['content_length'] = df['markdown_content'].str.len()

# Generate summary statistics
summary_stats = df['content_length'].describe()
print("\nSummary Statistics:")
print(summary_stats)

# Create histogram bins
bin_edges = np.arange(0, df['content_length'].max() + 500, 500)
hist_data = pd.cut(df['content_length'], bins=bin_edges)
hist_counts = hist_data.value_counts().sort_index()

print("\nLength Distribution:")
print(hist_counts)

# Plot histogram using matplotlib
plt.figure(figsize=(12, 6))
plt.hist(df['content_length'], bins=bin_edges, edgecolor='black')
plt.title('Distribution of Markdown Content Length')
plt.xlabel('Content Length (characters)')
plt.ylabel('Number of Files')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# Additional analysis
print("\nFiles with shortest content:")
print(df.nsmallest(5, 'content_length')[['filename', 'content_length']])

print("\nFiles with longest content:")
print(df.nlargest(5, 'content_length')[['filename', 'content_length']])