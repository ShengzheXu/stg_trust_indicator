import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Centers the embeddings by removing the mean (without scaling variance).

    Args:
        embeddings (np.ndarray): The input embeddings.

    Returns:
        np.ndarray: Centered embeddings.
    """
    return StandardScaler(with_std=False).fit_transform(embeddings)

def cluster_embeddings(data: np.ndarray, sample_size: int = 2000, print_silhouette: bool = False) -> tuple:
    """
    Flatten, center, sample, and cluster embeddings using KMeans, 
    choosing the best number of clusters based on silhouette score.

    Args:
        data (np.ndarray): Embeddings, either shape [batch, seq_len, hidden] or [num_tokens, hidden].
        sample_size (int): Max number of embeddings to use for clustering.

    Returns:
        Tuple:
            - best_k (int): Optimal number of clusters.
            - best_score (float): Best silhouette score achieved.
            - embedding_y (np.ndarray): Sampled and centered embeddings.
            - embedding_key (np.ndarray): Index mapping of sampled points.
            - embedding_cluster (np.ndarray): Cluster labels for sampled points.
    
    Example:
        embeddings = np.array(block_outputs[11].cpu()) # Assuming block_outputs[0] is the embedding tensor
        print(f"Embedding shape: {embeddings.shape}")
        best_k, best_sil, embedding_y, embedding_key, embedding_cluster = cluster_embeddings(embeddings, print_silhouette=True)
        print(f"Best k: {best_k}, Silhouette Score: {best_sil:.4f}")
    """
    if data.ndim == 3:
        flattened = data.reshape(-1, data.shape[-1])
        embedding_key = np.tile(np.arange(data.shape[0]), data.shape[1])
    elif data.ndim == 2:
        flattened = data
        embedding_key = np.zeros(data.shape[0])
    else:
        raise ValueError("Expected input with 2 or 3 dimensions")

    centered = center_embeddings(flattened)

    if centered.shape[0] > sample_size:
        sample_indices = np.random.choice(centered.shape[0], sample_size, replace=False)
        embedding_y = centered[sample_indices]
        embedding_key = embedding_key[sample_indices]
    else:
        embedding_y = centered

    silhouette_scores = []
    all_labels = []
    candidate_ks = range(2, 17)

    for k in candidate_ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        embedding_cluster = kmeans.fit_predict(embedding_y)
        # print(f"Clustered {embedding_y.shape} points into {len(np.unique(embedding_cluster))} clusters with k={k}")
        if len(np.unique(embedding_cluster)) < 2:
            print(f"Skipping k={k} due to insufficient clusters")
            continue
        score = silhouette_score(embedding_y, embedding_cluster)
        silhouette_scores.append(score)
        all_labels.append(embedding_cluster)

    if len(silhouette_scores) > 0 and max(silhouette_scores) >= 0.1:
        best_idx = int(np.argmax(silhouette_scores))
        best_k = candidate_ks[best_idx]
        best_score = silhouette_scores[best_idx]
        embedding_cluster = all_labels[best_idx]
    else:
        best_k = 1
        best_score = 0.0
        embedding_cluster = np.zeros(embedding_y.shape[0], dtype=int)
    
    if print_silhouette:
        print("Silhouette scores for candidate ks:")
        for k, score in zip(candidate_ks, silhouette_scores):
            print(f"k={k}: {score:.4f}")
    return best_k, best_score, embedding_y, embedding_key, embedding_cluster


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def inter_cosine(embeddings: np.ndarray, keys: np.ndarray = None, center: bool = True, sample_size: int = -1) -> float:
    """
    Compute average pairwise cosine similarity (inter) across all samples.

    Args:
        embeddings (np.ndarray): [n_samples, hidden_dim]
        keys (np.ndarray): Optional group IDs
        center (bool): Whether to center embeddings
        sample_size (int): If >= 0, sample one point per group

    Returns:
        float: Average cosine similarity between all vectors
    """
    if center:
        embeddings = StandardScaler(with_std=False).fit_transform(embeddings)

    if sample_size >= 0 and keys is not None:
        selected = []
        for k in np.unique(keys):
            group = embeddings[keys == k]
            if group.shape[0] < 1:
                continue
            idx = np.random.choice(len(group), min(sample_size, len(group)), replace=False)
            selected.append(group[idx])
        embeddings = np.vstack(selected)

    # cos = cosine_similarity(embeddings, embeddings)
    # avg_cos = ( np.sum(np.sum(cos)) - cos.shape[0] ) / 2 / ( cos.shape[0]*(cos.shape[0]-1) / 2 )
    sim_matrix = cosine_similarity(embeddings)
    upper_tri_sum = np.sum(sim_matrix) - embeddings.shape[0]
    num_pairs = embeddings.shape[0] * (embeddings.shape[0] - 1) / 2
    avg_cos = upper_tri_sum / (2 * num_pairs)
    return avg_cos
    

def intra_cosine(embeddings: np.ndarray, keys: np.ndarray, center: bool = True) -> float:
    """
    Compute average intra-group cosine similarity.

    Args:
        embeddings (np.ndarray): [n_samples, hidden_dim]
        keys (np.ndarray): Group identifiers
        center (bool): Whether to center embeddings within each group

    Returns:
        float: Mean of average intra-group cosine similarities
    """
    if center:
        embeddings = StandardScaler(with_std=False).fit_transform(embeddings)

    intra_scores = []
    for k in np.unique(keys):
        group = embeddings[keys == k]
        if group.shape[0] <= 1:
            continue
        if group.shape[0] > 1000:
            idx = np.random.choice(group.shape[0], 1000, replace=False)
            group = group[idx]
        score = inter_cosine(group)
        intra_scores.append(score)

    return np.mean(intra_scores) if intra_scores else 0.0


def compute_clustered_cosine_metrics(embedding_y: np.ndarray, 
                                     embedding_cluster: np.ndarray, 
                                     embedding_key: np.ndarray, 
                                     center: bool = True, 
                                     ignore_cluster: bool = False) -> tuple:
    """
    Compute inter- and intra-cluster cosine similarity metrics.

    Args:
        embedding_y (np.ndarray): Sampled & centered embeddings [n_samples, hidden_dim]
        embedding_cluster (np.ndarray): Cluster labels
        embedding_key (np.ndarray): Original group index for each point
        center (bool): Whether to center before similarity calculation
        ignore_cluster (bool): Treat all points as one cluster

    Returns:
        Tuple[float, float]: (inter_cluster_cosine, intra_cluster_cosine)

    Example:
        inter_cos, intra_cos = compute_clustered_cosine_metrics(
            embedding_y=embedding_y,
            embedding_cluster=embedding_cluster,
            embedding_key=embedding_key,
            center=True,
            ignore_cluster=False
        )
        print(f"Inter-cluster cosine similarity: {inter_cos:.4f}")
    """
    if ignore_cluster:
        embedding_cluster = np.zeros_like(embedding_cluster)

    inter_scores = []
    intra_scores = []

    for cluster_id in np.unique(embedding_cluster):
        cluster_embeddings = embedding_y[embedding_cluster == cluster_id]
        cluster_keys = embedding_key[embedding_cluster == cluster_id]

        inter = inter_cosine(cluster_embeddings, keys=cluster_keys, center=center, sample_size=1000)
        if not np.isnan(inter) and not np.isinf(inter):
            inter_scores.append(inter)

        intra = intra_cosine(cluster_embeddings, keys=cluster_keys, center=center)
        if not np.isnan(intra) and not np.isinf(intra):
            intra_scores.append(intra)

    inter_mean = np.mean(inter_scores) if inter_scores else 0.0
    intra_mean = np.mean(intra_scores) if intra_scores else 0.0

    return inter_mean, intra_mean


# if __name__ == "__main__":
#     print(f"Block 0: shape = {block_outputs[0].shape}")
#     for i, block_output in enumerate(block_outputs):
#         embeddings = np.array(block_output.cpu())
#         best_k, best_sil, embedding_y, embedding_key, embedding_cluster = cluster_embeddings(embeddings)
#         inter_cos, _ = compute_clustered_cosine_metrics(
#             embedding_y=embedding_y,
#             embedding_cluster=embedding_cluster,
#             embedding_key=embedding_key,
#             center=True,
#             ignore_cluster=False
#         )
#         print(f"Block {i} - Inter Cosine: {inter_cos:.4f}, Cluster Count: {best_k}, Silhouette: {best_sil:.4f}")