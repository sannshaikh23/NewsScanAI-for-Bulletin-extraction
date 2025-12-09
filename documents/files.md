# Core Project Files ("The Heart of the Project")

These files represent the central logic and essential components of the Video Search System.

## 1. The Brain (Search Logic)
*   **`retriever/retrieve.py`**
    *   **Why it's the heart**: This file contains the actual search algorithm. It takes a query image, converts it to a Fisher Vector, compares it against the entire index using cosine similarity, and returns the best matches. Without this, there is no search.

## 2. The Nervous System (Orchestration)
*   **`run_pipeline.py`**
    *   **Why it's the heart**: This script connects all the individual processing steps (keyframe extraction, SIFT extraction, GMM training, indexing) into a cohesive workflow. It ensures data flows correctly from raw video to searchable index.

## 3. The Face (User Interface)
*   **`web_app.py`**
    *   **Why it's the heart**: This is the entry point for the user. It runs the Flask server, handles image uploads, displays search results, and manages the video player. It bridges the gap between the complex backend algorithms and the user.

## 4. The Memory (Data Storage)
*   **`videosearch.db`**
    *   **Why it's the heart**: Stores the "truth" about which videos exist, their paths, and their indexing status.
*   **`work_dir/index.npy`**
    *   **Why it's the heart**: This IS the search index. It contains the mathematical representation (Fisher Vectors) of every frame in the system. If this file is lost, the system cannot search anything until re-indexing.

## 5. The Muscle (Processing)
*   **`indexer/global_descriptors/index_dataset.py`**
    *   **Why it's the heart**: This script does the heavy mathematical lifting of converting millions of local features into the compact global descriptors (Fisher Vectors) used for searching.
