# Project Algorithm and File Details

## Algorithms Used

### 1. Keyframe Extraction
*   **Tool**: FFmpeg
*   **Method**: Uniform sampling at 1 FPS (Frame Per Second).
*   **Purpose**: To reduce the video to a manageable set of representative images for processing.

### 2. Feature Extraction (SIFT)
*   **Algorithm**: Scale-Invariant Feature Transform (SIFT)
*   **Implementation**: OpenCV (`cv2.SIFT_create`)
*   **Purpose**: Detects and describes local features in images that are invariant to scale and rotation.
*   **Output**: Keypoints and 128-dimensional descriptor vectors.

### 3. Visual Vocabulary (GMM)
*   **Algorithm**: Gaussian Mixture Model (GMM)
*   **Implementation**: scikit-learn (`sklearn.mixture.GaussianMixture`)
*   **Configuration**: 256 components, diagonal covariance.
*   **Purpose**: To learn a statistical model of the visual feature space from a random sample of descriptors.

### 4. Image Encoding (Fisher Vectors)
*   **Algorithm**: Fisher Vector Encoding
*   **Purpose**: Encodes the set of local SIFT descriptors for an image into a single, fixed-length global descriptor (65,536 dimensions).
*   **Process**: Computes gradients of the log-likelihood with respect to GMM parameters, followed by power and L2 normalization.

### 5. Similarity Search
*   **Metric**: Cosine Similarity
*   **Implementation**: Dot product of L2-normalized Fisher Vectors.
*   **Purpose**: To rank indexed frames based on their visual similarity to the query image.

### 6. Adaptive Filtering
*   **Logic**:
    *   **Sample Images**: Uses a relaxed threshold (`max(0.3, top_score * 0.5)`).
    *   **Uploaded Images**: Uses a strict threshold (`top_score * 0.85`) to ensure high relevance.

---

## Important Files

### Core Application
*   **`web_app.py`**: The main Flask web application that handles HTTP requests, routing, and integrates the search logic with the UI.
*   **`run_pipeline.py`**: A command-line utility to run individual steps or the full indexing pipeline (extraction, training, indexing).

### Database & Data Management
*   **`videosearch.db`**: SQLite database storing video metadata.
*   **`db_utils.py`**: Utility functions for database interactions.
*   **`populate_db.py`**: Script to register new videos into the database.
*   **`process_videos.py`**: Automated script to process registered videos (extract features, index).

### Indexing & Retrieval Modules
*   **`indexer/keyframes/extract_keyframes.py`**: Wrapper around FFmpeg for extracting frames.
*   **`indexer/local_descriptors/extract_sift.py`**: Script for SIFT feature extraction.
*   **`indexer/global_descriptors/train_gmm.py`**: Script to train the Gaussian Mixture Model.
*   **`indexer/global_descriptors/index_dataset.py`**: Script to compute Fisher Vectors and create the search index.
*   **`retriever/retrieve.py`**: Core logic for searching the index using a query image.

### Web Interface
*   **`templates/index.html`**: The main search page template.
*   **`templates/video_player.html`**: The video playback page template.
