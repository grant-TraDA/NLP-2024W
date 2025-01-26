import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np
from collections import Counter
from nltk import bigrams, trigrams
from nltk import ngrams


def perform_eda(csv_file):
    # Output directory for EDA results
    output_dir = "outputs/eda"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    data = pd.read_csv(csv_file)

    # Ensure "tokens" exists and is properly formatted
    if "tokens" in data.columns:
        # Add a column for token counts
        data["token_counts"] = data["tokens"].apply(len)

        # Plot: Box Plot of Token Counts
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=data["token_counts"])
        plt.title("Box Plot of Token Counts")
        plt.xlabel("Token Count")
        plt.savefig(os.path.join(output_dir, "box_plot_token_counts.png"))
        plt.close()

        # Plot: Trimmed Histogram of Token Counts (95th Percentile)
        upper_limit = data["token_counts"].quantile(0.95)
        trimmed_data = data[data["token_counts"] <= upper_limit]
        plt.figure(figsize=(10, 5))
        sns.histplot(trimmed_data["token_counts"], bins=20, kde=True)
        plt.title("Distribution of Token Counts (Trimmed at 95th Percentile)")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, "trimmed_token_counts_distribution.png"))
        plt.close()

        # Plot: Density Plot of Token Counts
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data["token_counts"], fill=True)
        plt.title("Density Plot of Token Counts")
        plt.xlabel("Token Count")
        plt.ylabel("Density")
        plt.savefig(os.path.join(output_dir, "density_plot_token_counts.png"))
        plt.close()

        # Word Cloud of All Tokens
        all_tokens = [
            token
            for sublist in data["tokens"].apply(eval)
            if isinstance(sublist, list)
            for token in sublist
        ]
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            " ".join(all_tokens)
        )
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Tokens")
        plt.savefig(os.path.join(output_dir, "word_cloud_of_tokens.png"))
        plt.close()

        # Most Frequent Words
        token_counter = Counter(all_tokens)
        most_common_tokens = token_counter.most_common(20)
        tokens, counts = zip(*most_common_tokens)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(tokens), palette="husl")
        plt.title("Top 20 Most Frequent Tokens")
        plt.xlabel("Frequency")
        plt.ylabel("Tokens")
        plt.savefig(
            os.path.join(output_dir, "most_frequent_tokens.png"), bbox_inches="tight"
        )
        plt.close()

        # Generate Bigrams and Trigrams
        bigrams = list(ngrams(all_tokens, 2))
        trigrams = list(ngrams(all_tokens, 3))
        quadgrams = list(ngrams(all_tokens, 4))
        quintgrams = list(ngrams(all_tokens, 5))

        # Most Frequent Bigrams
        bigram_counter = Counter(bigrams)
        most_common_bigrams = bigram_counter.most_common(20)
        bigram_tokens, bigram_counts = zip(
            *[(" ".join(b), c) for b, c in most_common_bigrams]
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(bigram_counts), y=list(bigram_tokens), palette="coolwarm")
        plt.title("Top 20 Most Frequent Bigrams")
        plt.xlabel("Frequency")
        plt.ylabel("Bigrams")
        plt.savefig(
            os.path.join(output_dir, "most_frequent_bigrams.png"), bbox_inches="tight"
        )
        plt.close()

        # Most Frequent Trigrams
        trigram_counter = Counter(trigrams)
        most_common_trigrams = trigram_counter.most_common(20)
        trigram_tokens, trigram_counts = zip(
            *[(" ".join(t), c) for t, c in most_common_trigrams]
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(trigram_counts), y=list(trigram_tokens), palette="viridis")
        plt.title("Top 20 Most Frequent Trigrams")
        plt.xlabel("Frequency")
        plt.ylabel("Trigrams")
        plt.savefig(
            os.path.join(output_dir, "most_frequent_trigrams.png"), bbox_inches="tight"
        )
        plt.close()
        # Most Frequent quadgrams
        quadgram_counter = Counter(quadgrams)
        most_common_quadgrams = quadgram_counter.most_common(20)
        quadgram_tokens, quadgram_counts = zip(
            *[(" ".join(t), c) for t, c in most_common_quadgrams]
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(quadgram_counts), y=list(quadgram_tokens), palette="viridis")
        plt.title("Top 20 Most Frequent Quadgrams")
        plt.xlabel("Frequency")
        plt.ylabel("Trigrams")
        plt.savefig(
            os.path.join(output_dir, "most_frequent_quadgrams.png"), bbox_inches="tight"
        )
        plt.close()
        # Most Frequent quintgrams
        quintgram_counter = Counter(quintgrams)  # Replace `quadgrams` with `quintgrams`
        most_common_quintgrams = quintgram_counter.most_common(20)
        quintgram_tokens, quintgram_counts = zip(
            *[(" ".join(t), c) for t, c in most_common_quintgrams]
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=list(quintgram_counts), y=list(quintgram_tokens), palette="viridis"
        )
        plt.title("Top 20 Most Frequent Quintgrams")
        plt.xlabel("Frequency")
        plt.ylabel("Quintgrams")  # Updated from "Trigrams" to "Quintgrams"
        plt.savefig(
            os.path.join(output_dir, "most_frequent_quintgrams.png"),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    perform_eda("outputs/preprocessed_data_csv/preprocessed_agreements_all_states.csv")
