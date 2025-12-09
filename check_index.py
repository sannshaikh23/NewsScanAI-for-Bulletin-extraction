import numpy as np
import os

try:
    idx = np.load('work_dir/index.npy', allow_pickle=True).item()
    print(f"Total keys: {len(idx)}")
    print(f"Has sample2? {any('sample2' in k for k in idx)}")
    print(f"Has sample_video? {any('sample_video' in k for k in idx)}")
except Exception as e:
    print(f"Error: {e}")
