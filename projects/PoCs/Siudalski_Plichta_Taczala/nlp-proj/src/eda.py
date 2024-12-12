import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np
from collections import Counter
from nltk import bigrams,trigrams
from nltk.corpus import stopwords

def perform_eda(csv_file):
    output_dir = "outputs/eda"
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_file)

    if 'cleaned_text' in data.columns:
        data['text_length'] = data['cleaned_text'].apply(lambda x: len(str(x)))
        plt.figure(figsize=(12, 8))
        sns.histplot(data['text_length'], bins=20, kde=True)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
        plt.close()

    if 'tokens' in data.columns:
        data['token_count'] = data['tokens'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
        #Box Plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=data['token_count'])
        plt.title('Box Plot of Token Counts')
        plt.xlabel('Token Count')
        plt.savefig(os.path.join(output_dir, "box_plot_token_counts.png"))
        plt.close()

        #Trimed Outliers and Plot
        upper_limit = data['token_count'].quantile(0.95)  # 95th percentile
        trimmed_data = data[data['token_count'] <= upper_limit]
        plt.figure(figsize=(10, 5))
        sns.histplot(trimmed_data['token_count'], bins=20, kde=True)
        plt.title('Distribution of Token Counts (Trimmed)')
        plt.xlabel('Token Count (Trimmed at 95th Percentile)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, "trimmed_token_counts_distribution.png"))
        plt.close()

        #Density Plot
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data['token_count'], shade=True)
        plt.title('Density Plot of Token Counts')
        plt.xlabel('Token Count')
        plt.ylabel('Density')
        plt.savefig(os.path.join(output_dir, "density_plot_token_counts.png"))
        plt.close()

        all_tokens = [token for sublist in data['tokens'].apply(eval) if isinstance(sublist, list) for token in sublist]
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Tokens')
        plt.savefig(os.path.join(output_dir, "word_cloud_of_tokens.png"))
        plt.close()

        #Most Frequent Words
        token_counter = Counter(all_tokens)
        most_common_tokens = token_counter.most_common(20)
        tokens, counts = zip(*most_common_tokens)

        plt.figure(figsize=(10, 5))
        bar_plot = sns.barplot(x=list(counts), y=list(tokens), palette="husl")
        plt.title('Top 20 Most Frequent Tokens')
        plt.xlabel('Frequency')
        plt.ylabel('Tokens')

        for i, bar in enumerate(bar_plot.patches):
            bar.set_color(sns.color_palette("husl", len(bar_plot.patches))[i])

        plt.savefig(os.path.join(output_dir, "most_frequent_tokens.png"), bbox_inches='tight')
        plt.close()
        #Bigram
        bigram_list = [bigram for sublist in data['tokens'].apply(eval) if isinstance(sublist, list) for bigram in bigrams(sublist)]
        bigram_counter = Counter(bigram_list)
        most_common_bigrams = bigram_counter.most_common(20)
        bigram_terms, counts = zip(*most_common_bigrams)

        plt.figure(figsize=(10, 5))
        bigram_plot = sns.barplot(x=list(counts), y=[" ".join(bigram) for bigram in bigram_terms], palette="husl")
        plt.title('Top 20 Most Frequent Bigrams')
        plt.xlabel('Frequency')
        plt.ylabel('Bigrams')
        

        for i, bar in enumerate(bigram_plot.patches):
            bar.set_color(sns.color_palette("husl", len(bigram_plot.patches))[i])

        plt.savefig(os.path.join(output_dir, "most_frequent_bigrams.png"), bbox_inches='tight')
        plt.close()

        #Average Word Length
        data['avg_word_length'] = data['tokens'].apply(lambda x: np.mean([len(word) for word in eval(x)]) if isinstance(x, str) else 0)

        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data['avg_word_length'])
        plt.title('Average Word Length')
        plt.xlabel('Average Word Length')
        plt.savefig(os.path.join(output_dir, "box_plot_avg_word_length.png"), bbox_inches='tight')
        plt.close()

        #Trigram
        trigram_list = [trigram for sublist in data['tokens'].apply(lambda x: eval(x) if isinstance(x, str) else []) for trigram in trigrams(sublist)]
        trigram_counter = Counter(trigram_list)
        most_common_trigrams = trigram_counter.most_common(20)
        trigram_terms, counts = zip(*most_common_trigrams)

        plt.figure(figsize=(10, 5))
        trigram_plot = sns.barplot(x=list(counts), y=[" ".join(trigram) for trigram in trigram_terms], palette="husl")
        plt.title('Top 20 Most Frequent Trigrams', fontsize=16)
        plt.xlabel('Frequency', fontsize=14)
        plt.ylabel('Trigrams', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for i, bar in enumerate(trigram_plot.patches):
            bar.set_color(sns.color_palette("husl", len(trigram_plot.patches))[i])

        plt.savefig(os.path.join(output_dir, "most_frequent_trigrams.png"), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    perform_eda("outputs/preprocessed_data_csv/preprocessed_agreements.csv")
