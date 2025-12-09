import numpy as np
import argparse
import os
import pickle
import cv2
import sys

# Import functions from other scripts
# We need to add the indexer paths to sys.path or copy functions.
# For simplicity, I will assume the scripts are in known locations or importable.
# To make it robust, I'll duplicate the necessary logic or assume this script is run from root.
# Actually, let's just import if possible, or copy `compute_fisher_vector` logic.
# Since `compute_fisher_vector` is in `indexer/global_descriptors/index_dataset.py`, 
# I can try to import it if I set PYTHONPATH.
# Or I can just copy the function to a common utility or here. 
# Copying is safer to avoid path issues for the user.

def compute_fisher_vector(descriptors, gmm):
    """
    Computes the Fisher Vector for a set of descriptors.
    (Duplicated from index_dataset.py for standalone usage)
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(gmm.n_components * 2 * descriptors.shape[1], dtype=np.float32)
        
    q = gmm.predict_proba(descriptors)
    N = len(descriptors)
    D = descriptors.shape[1]
    means = gmm.means_
    covars = gmm.covariances_
    if gmm.covariance_type == 'diag':
        sigma = np.sqrt(covars)
        inv_sigma = 1.0 / sigma
    else:
        raise ValueError("Only diagonal covariance is supported for now.")
    weights = gmm.weights_
    Q = np.sum(q, axis=0)
    S = np.dot(q.T, descriptors)
    S_sq = np.dot(q.T, descriptors ** 2)
    G_mu = (S - Q[:, np.newaxis] * means) * inv_sigma
    G_mu = G_mu / np.sqrt(weights[:, np.newaxis])
    G_sigma = (S_sq - 2 * means * S + Q[:, np.newaxis] * (means ** 2))
    G_sigma = G_sigma * (inv_sigma ** 2) - Q[:, np.newaxis]
    G_sigma = G_sigma / np.sqrt(2 * weights[:, np.newaxis])
    fv = np.concatenate([G_mu.flatten(), G_sigma.flatten()])
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    norm = np.linalg.norm(fv)
    if norm > 0:
        fv = fv / norm
    return fv

def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve similar images")
    parser.add_argument("index_file", help="Path to the index file (.npy)")
    parser.add_argument("gmm_model", help="Path to the GMM model (.pickle)")
    parser.add_argument("query_image", help="Path to the query image")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Load Index
    print("Loading index...")
    index = np.load(args.index_file, allow_pickle=True).item()
    
    # Load GMM
    print("Loading GMM...")
    with open(args.gmm_model, 'rb') as f:
        gmm = pickle.load(f)
        
    # Process Query
    print(f"Processing query {args.query_image}...")
    descriptors = extract_sift_features(args.query_image)
    if descriptors is None:
        print("Error: Could not extract features from query image.")
        sys.exit(1)
        
    query_fv = compute_fisher_vector(descriptors, gmm)
    
    # Search
    print("Searching...")
    scores = []
    for name, fv in index.items():
        # Cosine similarity (since vectors are L2 normalized, it's just dot product)
        score = np.dot(query_fv, fv)
        scores.append((name, score))
        
    # Sort
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Print Results
    print(f"\nTop {args.top_k} Results:")
    for i in range(min(args.top_k, len(scores))):
        print(f"{i+1}. {scores[i][0]} (Score: {scores[i][1]:.4f})")
