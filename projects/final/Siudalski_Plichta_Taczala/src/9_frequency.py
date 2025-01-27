import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def count_target_words(df):
    target_words = ['always', 'rarely', 'often']
    results = []
    for idx, row in df.iterrows():
        counts = {word: row['tokens'].count(word) for word in target_words}
        counts['file_name'] = row['file_name']
        results.append(counts)
    return pd.DataFrame(results)

def plot_word_histogram(word_counts):
    words = ['always', 'rarely', 'often']
    frequencies = [word_counts[word].sum() for word in words]
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.title('Word Frequency Distribution')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add frequency labels on top of each bar
    for i, freq in enumerate(frequencies):
        plt.text(i, freq, str(freq), ha='center', va='bottom')
    plt.show()

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(input_file)
        print("Successfully loaded the CSV file.")
        print("\nFirst few rows of the data:")
        print(df.head())
        
        word_counts = count_target_words(df)
        plot_word_histogram(word_counts)
        
    except pd.errors.EmptyDataError:
        print(f"Error: '{input_file}' is empty.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()