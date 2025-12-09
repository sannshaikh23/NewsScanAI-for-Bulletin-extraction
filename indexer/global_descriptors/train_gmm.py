import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
import glob
import os
import pickle

def load_descriptors(files, sample_rate=1.0, max_descriptors=1000000):
    """
    Loads descriptors from a list of .npy files.
    
    Args:
        files (list): List of .npy files.
        sample_rate (float): Fraction of descriptors to load from each file.
        max_descriptors (int): Maximum total descriptors to load.
    """
    all_descriptors = []
    total_loaded = 0
    
    print(f"Loading descriptors from {len(files)} files...")
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True).item()
            des = data['descriptors']
            
            if des is None or len(des) == 0:
                continue
                
            if sample_rate < 1.0:
                n = len(des)
                n_sample = int(n * sample_rate)
                if n_sample > 0:
                    indices = np.random.choice(n, n_sample, replace=False)
                    des = des[indices]
            
            all_descriptors.append(des)
            total_loaded += len(des)
            
            if total_loaded >= max_descriptors:
                print(f"Reached maximum descriptors limit ({max_descriptors})")
                break
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not all_descriptors:
        return None
        
    return np.vstack(all_descriptors)

def train_gmm(descriptors, n_components=256, covariance_type='diag'):
    """
    Trains a GMM on the descriptors.
    """
    print(f"Training GMM with {n_components} components on {len(descriptors)} descriptors...")
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, verbose=1)
    gmm.fit(descriptors)
    print("GMM training converged.")
    return gmm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GMM on SIFT descriptors")
    parser.add_argument("input_dir", help="Directory containing .npy files with SIFT descriptors")
    parser.add_argument("output_model", help="Path to save the trained GMM model (pickle)")
    parser.add_argument("--k", type=int, default=256, help="Number of Gaussian components")
    parser.add_argument("--max_des", type=int, default=1000000, help="Max descriptors to use for training")
    
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.input_dir, "**", "*.npy"), recursive=True)
    if not files:
        print(f"No .npy files found in {args.input_dir}")
        sys.exit(1)
        
    descriptors = load_descriptors(files, max_descriptors=args.max_des)
    
    if descriptors is None:
        print("No descriptors loaded.")
        sys.exit(1)
        
    gmm = train_gmm(descriptors, n_components=args.k)
    
    # Save model
    with open(args.output_model, 'wb') as f:
        pickle.dump(gmm, f)
    
    print(f"Model saved to {args.output_model}")
