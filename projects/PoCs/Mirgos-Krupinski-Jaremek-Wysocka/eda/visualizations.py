import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# for single book

def plot_most_common_words(analysis, save_dir, top_n=10, show=False):
    """
    Plot a histogram of the most frequent words.

    Args:
        analysis (dict): Analysis dictionary from the TextAnalyzer class.
        top_n (int): Number of most common words to display.
    """
    most_common_words = analysis["most_common_words"][:top_n]
    words, counts = zip(*most_common_words)

    sns.barplot(x=counts, y=words, palette="viridis")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title(f"Most Frequent Words from {analysis['title']}")
    
    plt.savefig(f"{save_dir}/{analysis['title']}/common_words_{analysis['title']}", bbox_inches="tight")
    
    if show:
        plt.show()
    
    plt.close()
    



def plot_most_frequent_ngrams(analysis, save_dir, n=2, top_n=10, show=False):
    """
    Plot a histogram of the most frequent n-grams.

    Args:
        analysis (dict): Analysis dictionary from the TextAnalyzer class.
        n (int): Size of the n-grams to visualize (2 for bigrams, 3 for trigrams).
        top_n (int): Number of most frequent n-grams to display.
    """
    ngram_key = f"most_frequent_{'bigrams' if n == 2 else 'trigrams'}"
    most_frequent_ngrams = analysis[ngram_key][:top_n]
    ngrams, counts = zip(*most_frequent_ngrams)

    sns.barplot(x=counts, y=ngrams, palette="magma")
    plt.xlabel("Frequency")
    plt.ylabel("N-grams")
    plt.title(f"Most Frequent {n}-grams from {analysis['title']}")
    
    plt.savefig(f"{save_dir}/{analysis['title']}/frequent_{n}grams_{analysis['title']}", bbox_inches="tight")
    
    if show:
        plt.show()
    
    plt.close()


def plot_word_length_distribution(analysis, save_dir, show=False):
    """
    Plot the distribution of word lengths.

    Args:
        analysis (dict): Analysis dictionary from the TextAnalyzer class.
        save_dir (str): Directory to save the plot.
        show (bool): Whether to display the plot interactively.
    """
    word_lengths = analysis['word_length_distribution']

    sns.histplot(word_lengths, kde=True, bins=15, color="skyblue", kde_kws={"bw_adjust": 1.5})
    plt.xlabel("Word Length")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Word Lengths from {analysis['title']}")

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(f"{save_dir}/{analysis['title']}/word_length_{analysis['title']}", bbox_inches="tight")
    
    if show:
        plt.show()

    plt.close()


def plot_sentence_length_distribution(analysis, save_dir, show=False):
    """
    Plot the distribution of sentence lengths in words.

    Args:
        sentences (list): List of sentences from the TextAnalyzer class.
    """
    sentence_lengths = analysis['sentence_length_distribution']
    
    sns.histplot(sentence_lengths, kde=True, bins=15, color="coral")
    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Sentence Lengths from {analysis['title']}")
    plt.savefig(f"{save_dir}/{analysis['title']}/sentence_length_{analysis['title']}", bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


# for multiple books


def plot_bar_chart(results, metric, save_dir, title, xlabel=None, ylabel=None, color_palette=None, show=False):
    """
    Plot a sorted bar chart for a specific metric across multiple results with consistent colors for titles.
    
    Args:
        results (list of dict): List of analysis results from TextAnalyzer.
        metric (str): The key of the metric to plot (e.g., 'word_count', 'sentence_count').
        title (str, optional): Title of the plot. If None, a default title is generated.
        xlabel (str, optional): Label for the x-axis. If None, defaults to 'Books'.
        ylabel (str, optional): Label for the y-axis. If None, the metric name is used.
        ascending (bool, optional): If True, sort values in ascending order. Defaults to True.
        color_palette (dict, optional): A mapping of book names to specific colors. If None, colors are assigned dynamically.
    """
    data = [(res['title'], res[metric]) for res in results if metric in res]
    
    data = sorted(data, key=lambda x: x[1], reverse=True)
    
    names, metric_values = zip(*data)
    
    if color_palette is None:
        unique_names = set(names)
        color_palette = {name: color for name, color in zip(unique_names, sns.color_palette("tab10", len(unique_names)))}
    
    bar_colors = [color_palette[name] for name in names]

    sns.barplot(x=names, y=metric_values, palette=bar_colors)
    plt.xlabel(xlabel if xlabel else "Books")
    plt.ylabel(ylabel if ylabel else metric.replace("_", " ").title())
    plt.title(title if title else f"{metric.replace('_', ' ').title()} by Book")
    plt.xticks(rotation=45, ha="right")  # Rotate book names for readability
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/{metric}", bbox_inches="tight")
    
    if show:
        plt.show()

    plt.close()

    # Return the color palette for reuse
    return color_palette


