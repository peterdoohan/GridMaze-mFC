import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy.linalg as LA
from joblib import Parallel, delayed

# [Previous helper functions remain the same...]
def compute_direction_batch(neighborhood_batch):
    """Helper function for parallel direction computation"""
    directions = []
    for neighborhood in neighborhood_batch:
        centered = neighborhood - neighborhood.mean(axis=0)
        _, _, vh = LA.svd(centered, full_matrices=False)
        directions.append(vh[0])
    return np.array(directions)

def relabel_clusters(labels):
    """
    Reassign cluster labels to be continuous integers starting from 0,
    keeping -1 for noise points.
    
    Parameters:
    labels: np.array of cluster labels
    
    Returns:
    np.array of remapped labels
    """
    unique_labels = np.unique(labels)
    new_labels = labels.copy()
    
    # Keep -1 for noise, start regular labels from 0
    label_map = {-1: -1}  # Keep noise points as -1
    next_label = 0
    
    for label in unique_labels:
        if label != -1:
            label_map[label] = next_label
            next_label += 1
    
    # Vectorized label remapping
    for old_label, new_label in label_map.items():
        new_labels[labels == old_label] = new_label
    
    return new_labels

def cluster_curves(points, eps=0.1, min_samples=5, direction_weight=0.5, n_jobs=-1, batch_size=1000):
    """
    Optimized clustering for large 3D curve datasets with continuous labels.
    
    [Previous parameters documentation remains the same...]
    """
    n_points = len(points)
    k = min(15, n_points - 1)
    
    print("Computing nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=n_jobs).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    print("Computing local directions...")
    n_batches = (n_points + batch_size - 1) // batch_size
    neighborhood_batches = [points[indices[i:i+batch_size]] for i in range(0, n_points, batch_size)]
    
    directions_batches = Parallel(n_jobs=n_jobs)(
        delayed(compute_direction_batch)(batch) 
        for batch in neighborhood_batches
    )
    directions = np.vstack(directions_batches)
    directions = directions / LA.norm(directions, axis=1)[:, np.newaxis]
    
    print("Building similarity matrix...")
    k_sim = min(30, n_points)
    nbrs_sim = NearestNeighbors(n_neighbors=k_sim, algorithm='kd_tree', n_jobs=n_jobs).fit(points)
    distances_sim, indices_sim = nbrs_sim.kneighbors(points)
    
    rows = []
    cols = []
    vals = []
    
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_size_actual = batch_end - batch_start
        
        for i in range(batch_size_actual):
            idx = batch_start + i
            neighbors = indices_sim[idx]
            
            spatial_sim = np.exp(-distances_sim[idx]**2 / (2 * eps**2))
            dir_sim = np.abs(np.dot(directions[idx], directions[neighbors].T))
            
            combined_sim = (1 - direction_weight) * spatial_sim + direction_weight * dir_sim
            mask = combined_sim > 0.5
            
            rows.extend([idx] * np.sum(mask))
            cols.extend(neighbors[mask])
            vals.extend(combined_sim[mask])
    
    similarity = csr_matrix((vals, (rows, cols)), shape=(n_points, n_points))
    similarity = similarity.maximum(similarity.T)
    
    print("Finding connected components...")
    n_components, labels = connected_components(similarity, directed=False)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    small_clusters = unique_labels[counts < min_samples]
    if len(small_clusters) > 0:
        mask = np.isin(labels, small_clusters)
        labels[mask] = -1
    
    # Relabel clusters to ensure continuous labels
    labels = relabel_clusters(labels)
    
    return labels

def refine_clusters_batch(points, labels, smoothing_factor=0.1, n_jobs=-1, batch_size=1000):
    """
    Optimized cluster refinement for large datasets.
    """
    n_points = len(points)
    k = min(15, n_points - 1)
    
    print("Computing refinement neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=n_jobs).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    weights = np.exp(-distances**2 / (2 * smoothing_factor**2))
    refined_labels = labels.copy()
    
    print("Refining clusters...")
    for iteration in range(2):  # Reduced number of iterations for speed
        for batch_start in range(0, n_points, batch_size):
            batch_end = min(batch_start + batch_size, n_points)
            batch_indices = indices[batch_start:batch_end]
            batch_weights = weights[batch_start:batch_end]
            
            # Vectorized label voting
            batch_neighbor_labels = labels[batch_indices]
            batch_mask = labels[batch_start:batch_end] != -1
            
            for i in range(batch_end - batch_start):
                if not batch_mask[i]:
                    continue
                
                neighbor_labels = batch_neighbor_labels[i]
                neighbor_weights = batch_weights[i]
                
                unique_labels = np.unique(neighbor_labels[neighbor_labels != -1])
                if len(unique_labels) == 0:
                    continue
                
                label_votes = np.zeros(len(unique_labels))
                for j, label in enumerate(unique_labels):
                    label_votes[j] = np.sum(neighbor_weights[neighbor_labels == label])
                
                refined_labels[batch_start + i] = unique_labels[np.argmax(label_votes)]
    return relabel_clusters(refined_labels)

def cluster_large_dataset(points, eps=0.1, min_samples=5, direction_weight=0.5, n_jobs=-1):
    """
    Wrapper function with progress reporting and continuous labels.
    """
    print(f"Starting clustering of {len(points)} points...")
    
    labels = cluster_curves(points, eps=eps, min_samples=min_samples, 
                          direction_weight=direction_weight, n_jobs=n_jobs)
    
    print("Initial clustering complete. Starting refinement...")
    refined_labels = refine_clusters_batch(points, labels, n_jobs=n_jobs)
    
    # Final relabeling to ensure continuous labels
    final_labels = relabel_clusters(refined_labels)
    
    n_clusters = len(np.unique(final_labels)) - (1 if -1 in final_labels else 0)
    print(f"Clustering complete. Found {n_clusters} clusters.")
    
    return final_labels