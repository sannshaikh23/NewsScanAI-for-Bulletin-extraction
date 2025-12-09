# Video Search System - Project Details

## Overview
This is a content-based video retrieval system that allows users to search for similar scenes across multiple videos using a query image. The system extracts visual features from video frames and uses machine learning techniques to find the most similar frames.

## Technology Stack

### Programming Language
- **Python 3.x**: Core programming language for the entire system

### Web Framework
- **Flask**: Lightweight web framework for the search interface
  - Routes: `/`, `/search`, `/play_video`, `/get_sample_images`
  - Handles file uploads, JSON responses, and template rendering

### Computer Vision & Machine Learning Libraries

#### OpenCV (cv2)
- **Version**: Latest compatible with Python
- **Purpose**: 
  - Video processing and keyframe extraction
  - SIFT (Scale-Invariant Feature Transform) feature detection
  - Image I/O operations
- **Key Functions**:
  - `cv2.VideoCapture()`: Read video files
  - `cv2.SIFT_create()`: Create SIFT detector
  - `sift.detectAndCompute()`: Extract keypoints and descriptors

#### NumPy
- **Purpose**: 
  - Numerical computations
  - Array operations for feature vectors
  - Matrix operations for similarity scoring
- **Key Operations**:
  - `np.dot()`: Cosine similarity computation
  - `np.save()`/`np.load()`: Index persistence
  - Vector normalization

#### scikit-learn
- **Component**: `sklearn.mixture.GaussianMixture`
- **Purpose**: Train Gaussian Mixture Model for Fisher Vector encoding
- **Parameters**:
  - n_components = 256 (number of Gaussian components)
  - covariance_type = 'diag' (diagonal covariance matrices)

### Database
- **SQLite**: Lightweight database for video metadata
- **Tables**:
  - `videos`: Stores video information (name, path, indexed status, show_in_samples flag)
- **Python Interface**: `sqlite3` module

### Frontend Technologies
- **HTML5**: Structure
- **CSS3**: Styling with modern effects (flexbox, grid, transitions)
- **JavaScript (ES6)**: 
  - Fetch API for AJAX requests
  - DOM manipulation
  - SessionStorage for state persistence

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     Video Input                          │
│              (MP4 files in static/videos/)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Indexing Pipeline                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Extract    │→ │   Extract    │→ │    Train     │  │
│  │  Keyframes   │  │     SIFT     │  │     GMM      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                            │             │
│                                            ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Database   │← │    Index     │← │    Fisher    │  │
│  │   Update     │  │   Dataset    │  │    Vectors   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Search & Retrieval                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Query Image  │→ │    SIFT      │→ │    Fisher    │  │
│  │   Upload     │  │  Extraction  │  │    Vector    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                            │             │
│                                            ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Results    │← │   Ranking    │← │  Similarity  │  │
│  │   Display    │  │  & Filter    │  │  Scoring     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Algorithms & Techniques

### 1. Keyframe Extraction
- **Tool**: FFmpeg (via subprocess)
- **Method**: Uniform sampling at 1 FPS
- **Output**: JPEG images (6-digit frame numbers: 000001.jpg, 000002.jpg, ...)
- **Storage**: `work_dir/keyframes/<video_name>_keyframes/`

### 2. SIFT Feature Extraction
- **Algorithm**: Scale-Invariant Feature Transform (David Lowe, 1999)
- **Purpose**: Detect and describe local features invariant to scale and rotation
- **Output**: 
  - Keypoints: (x, y, size, angle)
  - Descriptors: 128-dimensional vectors
- **Storage**: `.npy` files in `work_dir/features/`
- **Naming Convention**: `<video>_keyframes_<frame>.npy`

### 3. Gaussian Mixture Model (GMM)
- **Purpose**: Learn a statistical model of the feature space
- **Training Data**: Random sample of SIFT descriptors from all videos
- **Parameters**:
  - K = 256 Gaussian components
  - Diagonal covariance matrices
  - EM algorithm for optimization
- **Storage**: `work_dir/gmm.pickle`

### 4. Fisher Vector Encoding
- **Algorithm**: Fisher Vector representation (Perronnin & Dance, 2007)
- **Process**:
  1. Compute soft assignments of descriptors to Gaussians (posterior probabilities)
  2. Calculate gradient statistics (mean and variance deviations)
  3. Power normalization: sign(x) * sqrt(|x|)
  4. L2 normalization
- **Dimensionality**: 2 * 256 * 128 = 65,536 dimensions
- **Advantages**:
  - Compact representation of variable-length features
  - Discriminative power
  - Suitable for dot product similarity

### 5. Indexing
- **Structure**: Python dictionary
  - Key: Frame identifier (e.g., "sample2_keyframes_000045")
  - Value: Fisher Vector (65,536-dim numpy array)
- **Storage**: `work_dir/index.npy` (pickle format)

### 6. Similarity Scoring
- **Metric**: Cosine similarity (dot product of L2-normalized vectors)
- **Formula**: `score = np.dot(query_fv, indexed_fv)`
- **Range**: [0, 1] where 1 = identical, 0 = completely different

### 7. Adaptive Filtering
- **Sample Images**: Relaxed threshold
  - Minimum score: max(0.3, top_score * 0.5)
  - Shows up to 10 results
- **Uploaded Images**: Strict threshold
  - Minimum score: top_score * 0.85
  - Filters unrelated results based on quality

## File Structure

```
videosearch-master/
├── web_app.py              # Flask application (main entry point)
├── db_utils.py             # Database operations
├── populate_db.py          # Script to register videos in database
├── process_videos.py       # Automated indexing pipeline
├── requirements.txt        # Python dependencies
├── videosearch.db          # SQLite database
├── indexer/
│   ├── keyframes/
│   │   └── extract_keyframes.py    # FFmpeg wrapper for keyframe extraction
│   ├── local_descriptors/
│   │   └── extract_sift.py         # SIFT feature extraction
│   └── global_descriptors/
│       ├── train_gmm.py            # GMM training
│       └── index_dataset.py        # Fisher Vector computation & indexing
├── retriever/
│   └── retrieve.py         # Search functionality (command-line)
├── templates/
│   ├── index.html          # Main search interface
│   └── video_player.html   # Video playback page
├── static/
│   └── videos/             # Video files for playback
├── work_dir/
│   ├── keyframes/          # Extracted frames
│   ├── features/           # SIFT descriptors (.npy)
│   ├── gmm.pickle          # Trained GMM model
│   └── index.npy           # Search index (Fisher Vectors)
└── .agent/
    └── workflows/          # Automation scripts
```

## Data Flow

### Indexing Workflow
1. **Input**: Video file (`.mp4`)
2. **Keyframe Extraction**: `extract_keyframes.py`
   - Input: Video path
   - Output: JPEG frames in `work_dir/keyframes/`
3. **Feature Extraction**: `extract_sift.py`
   - Input: Keyframe directory
   - Output: `.npy` files with SIFT descriptors
4. **GMM Training**: `train_gmm.py` (one-time)
   - Input: All SIFT descriptors
   - Output: `gmm.pickle`
5. **Indexing**: `index_dataset.py`
   - Input: SIFT descriptors + GMM
   - Output: Fisher Vectors in `index.npy`
6. **Database Update**: `populate_db.py` → `process_videos.py`
   - Registers video metadata
   - Marks as indexed

### Search Workflow
1. **User Upload**: Query image via web interface
2. **SIFT Extraction**: Extract descriptors from query
3. **Fisher Vector**: Encode using pre-trained GMM
4. **Similarity Scoring**: Compare with all indexed frames
5. **Ranking**: Sort by score (descending)
6. **Filtering**: Apply adaptive threshold
7. **Display**: Return top results with metadata

## Performance Characteristics

### Time Complexity
- **Indexing (per frame)**: O(N * K) where N = # SIFT features, K = 256 GMM components
- **Search**: O(M) where M = total indexed frames (~1000 frames for 5 videos)
- **Feature Extraction**: ~1-2 seconds per frame (SIFT)
- **Search Query**: ~0.1-0.5 seconds (in-memory comparison)

### Space Requirements
- **Per Frame**:
  - Keyframe JPEG: ~50-200 KB
  - SIFT descriptors: ~10-50 KB (.npy)
  - Fisher Vector: 256 KB (65,536 floats)
- **Total for 1000 frames**: ~300 MB index + ~200 MB features

## Key Features

### 1. Dual-Threshold Filtering
- Sample images: Show diverse results
- Uploaded images: Strict relevance filtering

### 2. SessionStorage Persistence
- Search results saved in browser
- Seamless back navigation from video player

### 3. Video Metadata Management
- SQLite database for scalability
- `show_in_samples` flag for UI control
- Indexing status tracking

### 4. Admin Workflow
- Add videos to `static/videos/`
- Run `populate_db.py` to register
- Run `process_videos.py` to index
- Automatic feature extraction and indexing

## Scalability Considerations

### Current Capacity
- Tested with: 7 videos, ~1000 total frames
- Search performance: <500ms per query

### Limitations
- In-memory index: Limited by RAM (~1GB for 10,000 frames)
- Sequential search: O(N) complexity

### Future Enhancements
1. **Approximate Nearest Neighbor (ANN)**:
   - FAISS, Annoy, or HNSW for sub-linear search
   - Trade-off: Slight accuracy for major speed gains
2. **Inverted File Index**:
   - Cluster Fisher Vectors
   - Search only relevant clusters
3. **Cloud Storage Integration**:
   - Move videos to S3/GCS
   - On-demand download for indexing
4. **Batch Processing**:
   - Parallel feature extraction
   - GPU acceleration for SIFT

## Dependencies

See `requirements.txt`:
- opencv-python
- numpy
- scikit-learn
- Flask
- Pillow

External:
- FFmpeg (system installation required)
