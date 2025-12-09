# News Scan AI for Bulletin Extraction - Complete Project Details

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [System Architecture](#system-architecture)
4. [Algorithms & Techniques](#algorithms--techniques)
5. [Complete Pipeline](#complete-pipeline)
6. [File Structure & Core Components](#file-structure--core-components)
7. [Data Flow](#data-flow)
8. [Cloud Integration (Cloudinary)](#cloud-integration-cloudinary)
9. [Performance & Scalability](#performance--scalability)
10. [Key Features](#key-features)
11. [Storage Architecture](#storage-architecture)

---

## Project Overview

**News Scan AI for Bulletin Extraction** is an intelligent AI-powered system designed for searching and extracting news bulletin segments from video archives. It enables journalists, editors, and researchers to quickly find specific news segments across large video databases by uploading a query image or selecting from sample frames.

The system uses advanced computer vision techniques including SIFT features, Fisher Vectors, and Gaussian Mixture Models to identify visually similar scenes and automatically navigate to the exact timestamp in the video.

---

## Technology Stack

### Programming Language
- **Python 3.7+**: Core programming language for the entire system

### Web Framework
- **Flask**: Lightweight web framework for the search interface
  - Routes: `/`, `/search`, `/play_video`, `/get_sample_images`
  - Handles file uploads, JSON responses, and template rendering

### Computer Vision & Machine Learning Libraries

#### OpenCV (cv2)
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
  - `videos`: Stores video information (name, path, indexed status, show_in_samples flag, cloudinary_url)
- **Python Interface**: `sqlite3` module

### Cloud Storage
- **Cloudinary**: CDN for video hosting and delivery
  - 25 GB free storage
  - Global CDN for fast delivery
  - Automatic video optimization

### Frontend Technologies
- **HTML5**: Structure
- **CSS3**: Styling with modern effects (flexbox, grid, gradients, transitions)
- **JavaScript (ES6)**: 
  - Fetch API for AJAX requests
  - DOM manipulation
  - SessionStorage for state persistence

### External Tools
- **FFmpeg**: Video processing and keyframe extraction

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Video Input                          │
│         (MP4 files - Local or Cloudinary CDN)            │
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

---

## Algorithms & Techniques

### 1. Keyframe Extraction
- **Tool**: FFmpeg (via subprocess)
- **Method**: Uniform sampling at 1 FPS (Frame Per Second)
- **Purpose**: Reduce video to manageable set of representative images
- **Output**: JPEG images (6-digit frame numbers: 000001.jpg, 000002.jpg, ...)
- **Storage**: `work_dir/keyframes/<video_name>_keyframes/`

### 2. SIFT Feature Extraction
- **Algorithm**: Scale-Invariant Feature Transform (David Lowe, 1999)
- **Implementation**: OpenCV (`cv2.SIFT_create`)
- **Purpose**: Detect and describe local features invariant to scale, rotation, and illumination
- **Output**: 
  - Keypoints: (x, y, size, angle)
  - Descriptors: 128-dimensional vectors
- **Storage**: `.npy` files in `work_dir/features/`
- **Naming Convention**: `<video>_keyframes_<frame>.npy`

### 3. Gaussian Mixture Model (GMM)
- **Algorithm**: Statistical model of visual feature space
- **Implementation**: scikit-learn (`sklearn.mixture.GaussianMixture`)
- **Training Data**: Random sample of SIFT descriptors from all videos
- **Parameters**:
  - K = 256 Gaussian components
  - Diagonal covariance matrices
  - EM algorithm for optimization
- **Storage**: `work_dir/gmm.pickle`
- **Purpose**: Learn "visual vocabulary" for encoding features

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
  - Key: Frame identifier (e.g., "test_video_1_keyframes_000045")
  - Value: Fisher Vector (65,536-dim numpy array)
- **Storage**: `work_dir/index.npy` (pickle format)
- **Purpose**: Searchable memory of all video frames

### 6. Similarity Scoring
- **Metric**: Cosine similarity (dot product of L2-normalized vectors)
- **Formula**: `score = np.dot(query_fv, indexed_fv)`
- **Range**: [0, 1] where 1 = identical, 0 = completely different

### 7. Adaptive Filtering
- **Sample Images**: Relaxed threshold
  - Minimum score: max(0.3, top_score * 0.5)
  - Shows up to 10 results
- **Uploaded Images**: Strict threshold
  - Absolute minimum: 0.50
  - Relative threshold: top_score * 0.85
  - Filters unrelated results based on quality
- **No Match Handling**: Displays user-friendly "⚠️ No Matching Videos Found" message

---

## Complete Pipeline

### Pipeline Overview

**Raw Video** → **Keyframes** → **Local Features (SIFT)** → **Global Descriptors (Fisher Vectors)** → **Search Index**

### Step-by-Step Flow

#### 1. Ingestion & Database (`process_videos.py`, `db_utils.py`)
- **Input**: Videos placed in `static/videos/` or uploaded to Cloudinary
- **Process**: System tracks videos in SQLite database (`videosearch.db`)
- **Output**: Database records updated with metadata and Cloudinary URLs

#### 2. Keyframe Extraction (`indexer/keyframes/extract_keyframes.py`)
- **Input**: Video file (e.g., `news_bulletin.mp4`)
- **Process**: Uses FFmpeg to extract frames at 1 frame/sec
- **Output**: JPEG images in `work_dir/keyframes/<video_name>_keyframes/`

#### 3. Feature Extraction (`indexer/local_descriptors/extract_sift.py`)
- **Input**: Keyframe JPEG images
- **Process**: Uses OpenCV to detect and compute SIFT descriptors
- **Output**: NumPy files (`.npy`) in `work_dir/features/`
- **Format**: Each file contains: `{'keypoints': [...], 'descriptors': [...]}`

#### 4. GMM Training (`indexer/global_descriptors/train_gmm.py`)
- **Input**: Random sample of SIFT descriptors from all videos
- **Process**: Train Gaussian Mixture Model with 256 components
- **Output**: `work_dir/gmm.pickle` (one-time training)

#### 5. Indexing (`indexer/global_descriptors/index_dataset.py`)
- **Input**: All `.npy` feature files + GMM model
- **Process**: Encode SIFT descriptors into Fisher Vectors
- **Output**: `work_dir/index.npy` (dictionary mapping frame_id → Fisher Vector)

#### 6. Search & Retrieval (`web_app.py`, `retriever/retrieve.py`)
- **Input**: User-uploaded query image
- **Process**:
  1. Extract SIFT features from query
  2. Compute Fisher Vector using GMM
  3. Calculate cosine similarity with all indexed frames
  4. Rank by score (descending)
  5. Apply adaptive threshold filtering
- **Output**: Ranked list of matching frames with metadata

---

## File Structure & Core Components

### Project Structure

```
news-scan-ai/
├── web_app.py              # Main Flask application (THE FACE)
├── db_utils.py             # Database helper functions
├── populate_db.py          # Register videos in database
├── process_videos.py       # Automated indexing pipeline (THE NERVOUS SYSTEM)
├── download_news_videos.py # Download & auto-upload to Cloudinary
├── run_pipeline.py         # Command-line pipeline orchestrator
├── requirements.txt        # Python dependencies
├── README.md               # Main documentation
├── videosearch.db          # SQLite database (THE MEMORY)
├── cloudinary_urls.json    # Video URL mappings
├── .env                    # Cloudinary credentials
├── documents/              # This documentation
│   └── DetailsofProject.md # Complete project details
├── templates/              # HTML templates
│   ├── index.html          # Main search interface
│   └── video_player.html   # Video playback page
├── static/
│   └── videos/             # Local video files (optional with Cloudinary)
├── work_dir/               # Generated files (THE MEMORY)
│   ├── keyframes/          # Extracted frames
│   ├── features/           # SIFT descriptors (.npy)
│   ├── gmm.pickle          # Trained Gaussian Mixture Model
│   └── index.npy           # Search index (Fisher Vectors)
└── indexer/                # Processing scripts (THE MUSCLE)
    ├── keyframes/
    │   └── extract_keyframes.py
    ├── local_descriptors/
    │   └── extract_sift.py
    └── global_descriptors/
        ├── train_gmm.py
        └── index_dataset.py (THE MUSCLE)
```

### Core Components (The Heart of the Project)

#### 1. The Brain (Search Logic)
- **`retriever/retrieve.py`**: Contains the actual search algorithm
- **Purpose**: Takes query image, converts to Fisher Vector, compares with index, returns best matches

#### 2. The Nervous System (Orchestration)
- **`run_pipeline.py`**: Connects all processing steps into cohesive workflow
- **`process_videos.py`**: Automated pipeline for new videos

#### 3. The Face (User Interface)
- **`web_app.py`**: Flask server, handles uploads, displays results, manages video player
- **`templates/index.html`**: Main search interface with modern UI

#### 4. The Memory (Data Storage)
- **`videosearch.db`**: Stores video metadata, paths, indexed status, Cloudinary URLs
- **`work_dir/index.npy`**: THE search index containing Fisher Vectors
- **`work_dir/gmm.pickle`**: Visual vocabulary for encoding

#### 5. The Muscle (Processing)
- **`indexer/global_descriptors/index_dataset.py`**: Heavy mathematical lifting
- **Purpose**: Converts millions of local features into compact global descriptors

---

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
6. **Filtering**: Apply adaptive threshold (0.50 minimum for uploads)
7. **Display**: Return top results with metadata

---

## Cloud Integration (Cloudinary)

### Integration Status

✅ **ALL VIDEOS CONFIGURED FOR CLOUDINARY**
- Total videos: 24
- With Cloudinary URLs: 24
- Without Cloudinary URLs: 0

### How It Works

1. **Video Upload**: Videos uploaded to Cloudinary CDN
2. **Database Storage**: Cloudinary URLs stored in `videosearch.db`
3. **Video Playback**: Videos stream directly from Cloudinary to browser
4. **Local Backup**: 2 sample videos kept in `static/videos/` for reference

### Code Changes

| File | Change | Purpose |
|------|--------|---------|
| `video_player.html` | Uses `{{ video_url }}` | Dynamic URL from database |
| `db_utils.py` | Added `get_video_url()` | Returns Cloudinary URL |
| `web_app.py` | Updated `play_video()` | Fetches URL from database |
| `download_news_videos.py` | Auto-uploads to Cloudinary | Automatic workflow |

### Auto-Upload Workflow

**New Feature**: `download_news_videos.py` now automatically:
1. Downloads videos from YouTube
2. Uploads to Cloudinary
3. Adds to database with Cloudinary URL
4. Updates `cloudinary_urls.json` mapping
5. Ready to search immediately!

### Internet Requirements

| Component | Requires Internet? | Reason |
|-----------|-------------------|--------|
| **Video Playback** | ✅ YES | Videos on Cloudinary CDN |
| **Download New Videos** | ✅ YES | Downloads from YouTube |
| **Video Search** | ❌ NO | Local index and features |
| **Upload Query Image** | ❌ NO | Processed locally |
| **View Search Results** | ❌ NO | Thumbnails from local keyframes |

### Benefits Achieved

- 🚀 Faster video delivery via global CDN
- 💾 ~2-3 GB disk space saved locally
- ☁️ 25 GB free Cloudinary storage utilized
- 🔒 Credentials secured in `.env` file
- 🌍 Better performance for remote users

---

## Performance & Scalability

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

### Current Capacity
- Tested with: 24 videos, ~1000+ total frames
- Search performance: <500ms per query
- In-memory index: Limited by RAM (~1GB for 10,000 frames)

### Scalability Limitations
- In-memory index: Limited by RAM
- Sequential search: O(N) complexity

### Future Enhancements
1. **Approximate Nearest Neighbor (ANN)**:
   - FAISS, Annoy, or HNSW for sub-linear search
   - Trade-off: Slight accuracy for major speed gains
2. **Inverted File Index**:
   - Cluster Fisher Vectors
   - Search only relevant clusters
3. **Cloud Storage Integration**:
   - Already implemented with Cloudinary
4. **Batch Processing**:
   - Parallel feature extraction
   - GPU acceleration for SIFT

---

## Key Features

### 1. Dual-Threshold Filtering
- **Sample Images**: Relaxed threshold (0.3 minimum) for diverse results
- **Uploaded Images**: Strict threshold (0.50 minimum) for high relevance
- **No Match Handling**: User-friendly warning message with professional UI

### 2. SessionStorage Persistence
- Search results saved in browser
- Seamless back navigation from video player
- Preserves query state

### 3. Video Metadata Management
- SQLite database for scalability
- `show_in_samples` flag for UI control
- Indexing status tracking
- Cloudinary URL storage

### 4. Admin Workflow
- Add videos to `static/videos/` or upload to Cloudinary
- Run `populate_db.py` to register
- Run `process_videos.py` to index
- Automatic feature extraction and indexing

### 5. Modern UI
- Clean, responsive web interface
- Professional gradient backgrounds
- Smooth transitions and hover effects
- User-friendly error messages

---

## Storage Architecture

### Database vs File Storage

**Videos (File System Storage)**:
- Actual video files (`.mp4`) stored as files
- Location: Cloudinary CDN (online) or `static/videos/` (local)
- **NOT** stored in database (bad practice for large files)

**Metadata (Database Storage)**:
- `videosearch.db` stores references and metadata
- Contains: filename, path, indexed status, Cloudinary URL
- Points to files but doesn't contain them

### Why This Architecture?

✅ **Separation of Concerns**:
- Database: Fast queries for metadata
- File System/CDN: Efficient storage and delivery of large files

✅ **Scalability**:
- Database stays small and fast
- Videos can be moved to cloud without database changes

✅ **Performance**:
- CDN delivers videos globally with low latency
- Local index enables fast search

### Cloud Storage Recommendations

**For Video Files**:
1. **Cloudinary** ⭐ (RECOMMENDED - Currently Implemented)
   - Free: 25 GB storage + 25 GB bandwidth/month
   - Features: CDN, automatic optimization, direct streaming
2. **Backblaze B2** (S3 Alternative)
   - Free: 10 GB storage + 1 GB daily download
3. **Firebase Storage**
   - Free: 5 GB storage + 1 GB/day download

**For Metadata**:
- **SQLite** (Current): Perfect for local/small deployments
- **PostgreSQL/Neon**: For cloud deployments or scaling
- **PostgreSQL with pgvector**: Can replace `index.npy` for vector search

**NOT Recommended**:
- ❌ Storing videos in database (bloats DB, slow queries)
- ❌ Storing keyframes/features in cloud (local access faster)
- ❌ GitHub LFS for video streaming (not designed for this)

---

## Dependencies

See `requirements.txt`:
- opencv-python
- numpy
- scikit-learn
- Flask
- Pillow
- cloudinary (optional)

External:
- FFmpeg (system installation required)

---

## Summary

**News Scan AI for Bulletin Extraction** is a production-ready, intelligent video search system that combines:
- Advanced computer vision (SIFT, Fisher Vectors)
- Cloud infrastructure (Cloudinary CDN)
- Modern web interface (Flask, HTML5, CSS3)
- Efficient algorithms (GMM, cosine similarity)
- Smart filtering (adaptive thresholds)

The system is designed for news organizations to quickly search and extract bulletin segments from large video archives, with a focus on accuracy, performance, and user experience.
