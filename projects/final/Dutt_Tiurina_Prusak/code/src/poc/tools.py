import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.metrics import silhouette_score
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Function to generate embeddings
def generate_embeddings(texts, tokenizer, model, batch_size=16):
    """
    Generate embeddings for a list of texts using DistilRoBERTa with mean pooling.
    """
    embeddings = []
    # Wrap the loop with tqdm for a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings", unit="batch"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling for sentence representation
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

            # Compute mean of embeddings, taking into account the attention mask
            masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, hidden_size)
            sum_embeddings = masked_embeddings.sum(dim=1)  # Shape: (batch_size, hidden_size)
            count_embeddings = attention_mask.sum(dim=1)  # Shape: (batch_size)

            # Avoid division by zero and compute mean
            mean_embeddings = sum_embeddings / count_embeddings.unsqueeze(-1).clamp(min=1e-9)  # Shape: (batch_size, hidden_size)
            embeddings.append(mean_embeddings)
    return torch.cat(embeddings, dim=0)

def reduce_dimensionality(embeddings, n_components=50, algo='none'):
    """
    Reduce the dimensionality of embeddings using PCA.
    """
    if algo == 'pca':
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    elif algo == 'umap':
        reducer = umap.UMAP(n_components=n_components)
        reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())
    elif algo == 'none':
        reduced_embeddings = embeddings.cpu().numpy()
    return reduced_embeddings

def perform_clustering(embeddings, n_clusters=5, algo='kmeans'):
    """
    Cluster embeddings using specified algorithm ('kmeans', 'hdbscan', 'agg', 'spectral', 'gmm', 'dbscan').
    Embeddings are reduced via `reduce_dimensionality`.

    Params:
        embeddings (array): Data to cluster.
        n_clusters (int): Number of clusters (used in relevant methods).
        algo (str): Clustering algorithm.
    Returns:
        clusters (array): Cluster labels (-1 for noise in some methods).
    """
    reduced_embeddings = reduce_dimensionality(embeddings)

    if algo == 'kmeans':
        clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(reduced_embeddings)
    elif algo == 'hdbscan':
        clusters = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(reduced_embeddings)
    elif algo == 'agg':
        clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(reduced_embeddings)
    elif algo == 'spectral':
        clusters = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42).fit_predict(reduced_embeddings)
    elif algo == 'gmm':
        clusters = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(reduced_embeddings)
    elif algo == 'dbscan':
        clusters = DBSCAN(eps=0.5, min_samples=10).fit_predict(reduced_embeddings)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    return clusters


def tokenize(text):
    """
    Tokenize single text example:
        remove punctruation and lowercase
        split by spaces
        remove stopwords
    """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    tokens = text.split()  # Simple split by spaces
    tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return tokens

def calc_cluster_metrics(clusters, embeddings):
    """
    Calculate result metrics for predicted clusters:
        number of clusters
        silhoette score
    """
    unique_clusters = set(clusters)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    num_clusters = len(unique_clusters)
    sil_score = -1
    if (num_clusters > 1): # silhoette score requires at least 2 clusters
        non_noise_mask = clusters != -1
        filtered_embeddings = embeddings[non_noise_mask]
        filtered_labels = clusters[non_noise_mask]
        sil_score = silhouette_score(filtered_embeddings, filtered_labels, metric='euclidean')
    return [num_clusters, sil_score]


