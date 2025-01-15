from collections import Counter
import re
import spacy
import stopwordsiso as stopwords_iso


class TextAnalyzer:
    def __init__(self, text):
        """
        Initialize the TextAnalyzer with preprocessed text.
        
        Args:
            text (str): The raw text to analyze.
        """
        self.nlp = spacy.load("pl_core_news_sm")
        self.stop_words = stopwords_iso.stopwords("pl")
        
        self.original_text = text
        self.preprocessed_text = self._preprocess_text(text).split()
        self.sentences = self._split_into_sentences()
    
    # def _preprocess_text(self, text):
    #     """Lowercase, tokenize, lemmatize, and remove non-alphabetic tokens and stopwords."""
    #     doc = self.nlp(text.lower())  # Ensure Spacy processes the text
    #     tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stop_words]
    #     return " ".join(tokens)
    
    def _preprocess_text(self, text):
        """
        Preprocess long text by splitting it into manageable chunks,
        lowercasing, tokenizing, lemmatizing, and removing non-alphabetic tokens and stopwords.
        """
        # Increase SpaCy's maximum length limit
        # self.nlp.max_length = max(len(text), 1_000_000)  # Adjust to accommodate larger texts

        # Split the text into chunks
        chunk_size = 100_000  # Process in chunks of 100,000 characters
        tokens = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = self.nlp(chunk.lower())  # Process chunk with SpaCy
            tokens.extend(token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stop_words)

        return " ".join(tokens)

    
    def _split_into_sentences(self):
        """Split the original text into sentences based on punctuation marks."""
        return re.split(r'[.!?]', self.original_text)
    
    def word_count(self):
        """Return the total word count of preprocessed words."""
        return len(self.original_text.split())  # Use split to count words safely
    
    def word_length_distribution(self):
        """
        Return the distribution of word lengths in the preprocessed text.

        Returns:
            list: List of integers representing the length of each word.
        """
        chunk_size = 100_000
        word_lengths = []
        for i in range(0, len(self.original_text), chunk_size):
            chunk = self.original_text[i:i + chunk_size]
            doc = self.nlp(chunk.lower())
            tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stop_words]
            word_lengths.extend(len(word) for word in tokens)
        
        return word_lengths

    
    def most_common_words(self, top_n=10):
        """
        Return the most common words in the preprocessed text.
        
        Args:
            top_n (int): Number of top common words to return.

        Returns:
            list: List of tuples with words and their counts.
        """
        chunk_size = 100_000
        counter = Counter()
        for i in range(0, len(self.original_text), chunk_size):
            chunk = self.original_text[i:i + chunk_size]
            # Tokenize and preprocess this chunk
            doc = self.nlp(chunk.lower())
            tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stop_words]
            counter.update(tokens)
        
        return counter.most_common(top_n)


    def sentence_count(self):
        """Return the total sentence count from the original text."""
        return len([s for s in self.sentences if s.strip()])  # Exclude empty sentences
    
    def sentence_length_distribution(self):
        """
        Return the distribution of sentence lengths in words.
        
        Returns:
            list: List of integers representing the word count in each sentence.
        """
        return [len(sentence.split()) for sentence in self.sentences if sentence.strip()]


    def mean_word_length(self):
        """Return the mean word length of words in the original text."""
        words = self.original_text.split()
        total_length = sum(len(word) for word in words)
        return total_length / len(words) if words else 0

    def mean_sentence_length(self):
        """Return the mean sentence length (in words) based on the preprocessed text."""
        sentence_lengths = [len(sentence.split()) for sentence in self.sentences if sentence.strip()]
        return sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    def most_frequent_ngrams(self, n=2, top_n=10):
        """
        Find the most frequent n-grams in the tokenized original text.

        Args:
            n (int): Size of the n-grams (e.g., 2 for bigrams, 3 for trigrams).
            top_n (int): Number of top n-grams to return.

        Returns:
            list: List of tuples with the n-grams and their counts.
        """
        
        chunk_size = 100_000
        tokens = []
        for i in range(0, len(self.original_text), chunk_size):
            chunk = self.original_text[i:i + chunk_size]
            doc = self.nlp(chunk.lower())
            tokens_batch = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stop_words]
            tokens.extend(tokens_batch)
            

        if len(tokens) < n:
            return []

        # Generate n-grams
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngram_counts = Counter([" ".join(ngram) for ngram in ngrams])

        return ngram_counts.most_common(top_n)

    def analyze(self):
        """
        Perform a comprehensive analysis of the text.

        Returns:
            dict: Dictionary containing analysis results.
        """
        analysis = {
            "word_count": self.word_count(),
            "sentence_count": self.sentence_count(),
            "most_common_words": self.most_common_words(),
            "mean_word_length": self.mean_word_length(),
            "mean_sentence_length": self.mean_sentence_length(),
            "sentence_length_distribution": self.sentence_length_distribution(),
            "word_length_distribution": self.word_length_distribution(),
            "most_frequent_bigrams": self.most_frequent_ngrams(2),
            "most_frequent_trigrams": self.most_frequent_ngrams(3),
        }

        return analysis


