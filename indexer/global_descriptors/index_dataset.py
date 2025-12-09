import numpy as np
import pickle
import argparse
import glob
import os
import sys
from sklearn.mixture import GaussianMixture

# Force UTF-8 for stdout to handle Hindi filenames
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def compute_fisher_vector(descriptors, gmm):
    """
    Computes the Fisher Vector for a set of descriptors.
    
    Args:
        descriptors (np.array): N x D array of descriptors.
        gmm (sklearn.mixture.GaussianMixture): Trained GMM.
        
    Returns:
        fv (np.array): Fisher Vector.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(gmm.n_components * 2 * descriptors.shape[1], dtype=np.float32)
        
    # 1. Compute posterior probabilities (soft assignments)
    # N x K
    q = gmm.predict_proba(descriptors)
    
    # 2. Compute statistics
    # K
    N = len(descriptors)
    
    # D
    D = descriptors.shape[1]
    
    # K x D
    means = gmm.means_
    covars = gmm.covariances_
    if gmm.covariance_type == 'diag':
        # covars is K x D
        sigma = np.sqrt(covars)
        inv_sigma = 1.0 / sigma
    else:
        raise ValueError("Only diagonal covariance is supported for now.")
        
    weights = gmm.weights_
    
    # Accumulate statistics
    # Q_k = sum(q_ik)
    Q = np.sum(q, axis=0) # Shape (K,)
    
    # S_k = sum(q_ik * x_i)
    S = np.dot(q.T, descriptors) # Shape (K, D)
    
    # S_sq_k = sum(q_ik * x_i^2)
    S_sq = np.dot(q.T, descriptors ** 2) # Shape (K, D)
    
    # 3. Compute gradients (Fisher Vector components)
    # Gradient w.r.t mean
    # d_mu_k = (S_k - Q_k * mu_k) / (sqrt(w_k) * sigma_k)
    # Note: standard FV often ignores sqrt(w_k) or handles it differently. 
    # We follow the "Improved Fisher Vector" formulation usually.
    # Let's use the standard formulation:
    # G_mu_k = 1/sqrt(w_k) * sum(q_ik * (x_i - mu_k) / sigma_k)
    #        = 1/sqrt(w_k) * (S_k - Q_k * mu_k) / sigma_k
    
    G_mu = (S - Q[:, np.newaxis] * means) * inv_sigma
    G_mu = G_mu / np.sqrt(weights[:, np.newaxis])
    
    # Gradient w.r.t sigma
    # G_sigma_k = 1/sqrt(2*w_k) * sum(q_ik * ((x_i - mu_k)^2 / sigma_k^2 - 1))
    #           = 1/sqrt(2*w_k) * (S_sq_k - 2*mu_k*S_k + Q_k*mu_k^2 - Q_k*sigma_k^2) / sigma_k^2
    # Simplified: (S_sq - 2*mu*S + Q*mu^2) / sigma^2 - Q
    
    G_sigma = (S_sq - 2 * means * S + Q[:, np.newaxis] * (means ** 2))
    G_sigma = G_sigma * (inv_sigma ** 2) - Q[:, np.newaxis]
    G_sigma = G_sigma / np.sqrt(2 * weights[:, np.newaxis])
    
    # Concatenate
    fv = np.concatenate([G_mu.flatten(), G_sigma.flatten()])
    
    # 4. Power Normalization
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    
    # 5. L2 Normalization
    norm = np.linalg.norm(fv)
    if norm > 0:
        fv = fv / norm
        
    return fv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index dataset using Fisher Vectors")
    parser.add_argument("input_dir", help="Directory containing .npy files with SIFT descriptors")
    parser.add_argument("gmm_model", help="Path to trained GMM model (pickle)")
    parser.add_argument("output_index", help="Path to save the index (npy)")
    
    args = parser.parse_args()
    
    # Load GMM
    with open(args.gmm_model, 'rb') as f:
        gmm = pickle.load(f)
        
    files = glob.glob(os.path.join(args.input_dir, "**", "*.npy"), recursive=True)
    files.sort()
    
    index_data = {}
    
    print(f"Indexing {len(files)} files...")
    
    for i, f in enumerate(files):
        try:
            name = os.path.splitext(os.path.basename(f))[0]
            data = np.load(f, allow_pickle=True).item()
            des = data['descriptors']
            
            fv = compute_fisher_vector(des, gmm)
            index_data[name] = fv
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(files)}")
                
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # Save index
    np.save(args.output_index, index_data)
    print(f"Index saved to {args.output_index}")
