# Video Search System

A content-based video retrieval system that enables searching for similar scenes across multiple videos using image queries. Built with Python, Flask, and computer vision techniques.

## Features

- 🔍 **Image-based Search**: Upload any image to find similar scenes in your video database
- 🎬 **Video Playback**: Click on results to play the video at the matching timestamp
- 📊 **Sample Queries**: Pre-loaded sample images for quick testing
- 🎯 **Smart Filtering**: Adaptive threshold system to show only relevant results
- 💾 **Metadata Management**: SQLite database for efficient video organization
- ⚡ **Fast Retrieval**: Fisher Vector encoding for efficient similarity search

## Technology Stack

- **Backend**: Python 3.x, Flask
- **Computer Vision**: OpenCV (SIFT features)
- **Machine Learning**: scikit-learn (Gaussian Mixture Models, Fisher Vectors)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript (ES6)
- **Video Processing**: FFmpeg

## Prerequisites

### System Requirements
- Python 3.7 or higher
- FFmpeg (for video processing)

### Install FFmpeg on Windows
```powershell
winget install FFmpeg
```

Or download from: https://ffmpeg.org/download.html

## Installation

### 1. Clone or Download the Project
```bash
cd videosearch-master
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- opencv-python
- numpy
- scikit-learn
- Flask
- Pillow

## Usage

### Quick Start (Using Existing Index)

If the project already has indexed videos in `work_dir/`, you can start the web app immediately:

```bash
python web_app.py
```

Then open your browser to: `http://127.0.0.1:5000`

### Adding New Videos (Admin Workflow)

#### Step 1: Add Video Files
Place your `.mp4` video files in the `static/videos/` directory:

```bash
# Example:
static/videos/my_video.mp4
```

#### Step 2: Register Videos in Database
```bash
python populate_db.py
```

This scans `static/videos/` and adds new videos to the database.

#### Step 3: Process & Index Videos
```bash
python process_videos.py
```

This command will:
- Extract keyframes (1 frame per second)
- Extract SIFT features from each frame
- Encode features as Fisher Vectors
- Update the search index
- Mark videos as "indexed" in the database

**⏱️ Processing Time**: ~2-5 minutes per video (depending on length)

#### Step 4: Restart the Web App
```bash
# Stop the running app (Ctrl+C)
python web_app.py
```

The new videos are now searchable!

### First-Time Setup (No Existing Index)

If starting from scratch with new videos:

```bash
# 1. Add videos to static/videos/

# 2. Register in database
python populate_db.py

# 3. Index the videos
python process_videos.py

# 4. Start the web app
python web_app.py
```

## How to Use the Search Interface

### Option 1: Upload Query Image
1. Click **"Upload Query Image"**
2. Select an image file from your computer
3. View results ranked by similarity
4. Click any result to play the video at that timestamp

### Option 2: Use Sample Images
1. Click **"Load Sample Images"**
2. Click on any sample thumbnail
3. View results and play videos

### Navigation
- **Back Button**: Returns to search results (preserves your query)
- **Replay Button**: Restart video from the matched frame

## Project Structure

```
videosearch-master/
├── web_app.py              # Main Flask application
├── db_utils.py             # Database helper functions
├── populate_db.py          # Register videos in database
├── process_videos.py       # Automated indexing pipeline
├── requirements.txt        # Python dependencies
├── README2.md              # This file
├── PROJECT_DETAILS.md      # Technical documentation
├── templates/              # HTML templates
│   ├── index.html
│   └── video_player.html
├── static/
│   └── videos/             # Video files (MP4)
├── work_dir/               # Generated files
│   ├── keyframes/          # Extracted frames
│   ├── features/           # SIFT descriptors
│   ├── gmm.pickle          # Trained model
│   └── index.npy           # Search index
└── indexer/                # Processing scripts
    ├── keyframes/
    ├── local_descriptors/
    └── global_descriptors/
```

## Terminal Commands Reference

### Web Application
```bash
# Start the web server
python web_app.py

# Access the application
# Open browser: http://127.0.0.1:5000

# Stop the server
# Press Ctrl+C in the terminal
```

### Video Management
```bash
# Add new videos to database
python populate_db.py

# Process pending videos (extract + index)
python process_videos.py

# Check database contents
python -c "from db_utils import get_all_videos; print([v['name'] for v in get_all_videos()])"
```

### Manual Processing (Advanced)
```bash
# Extract keyframes only
python indexer/keyframes/extract_keyframes.py video.mp4 output_dir 1

# Extract SIFT features only
python indexer/local_descriptors/extract_sift.py keyframes_dir features_dir

# Train GMM (first time only)
python indexer/global_descriptors/train_gmm.py features_dir work_dir/gmm.pickle --k 256

# Build search index
python indexer/global_descriptors/index_dataset.py features_dir work_dir/gmm.pickle work_dir/index.npy
```

## Troubleshooting

### Videos Not Playing
- Ensure videos are in `static/videos/` directory
- Check video format (MP4 recommended)
- Verify FFmpeg is installed: `ffmpeg -version`

### Search Returns No Results
- Run `process_videos.py` to ensure videos are indexed
- Check `work_dir/index.npy` exists
- Restart the web app to reload the index

### "Module not found" Errors
```bash
pip install -r requirements.txt
```

### FFmpeg Not Found
- Install FFmpeg and add to system PATH
- Verify installation: `ffmpeg -version`

## Performance Tips

- **Faster Indexing**: Process videos in batches during off-hours
- **Better Results**: Use high-quality query images
- **Reduce Index Size**: Index only key scenes (modify frame rate)

## Limitations

- **Index Loading**: All Fisher Vectors loaded in RAM (limit: ~10,000 frames)
- **File Format**: Only MP4 videos supported for playback
- **Processing Time**: ~2-5 minutes per video for indexing

## Future Enhancements

- [ ] Approximate Nearest Neighbor search (FAISS)
- [ ] Cloud storage integration (AWS S3, Google Cloud)
- [ ] Admin UI for video management
- [ ] Batch upload support
- [ ] Video thumbnail generation

## Technical Details

For in-depth technical documentation, see [PROJECT_DETAILS.md](PROJECT_DETAILS.md)

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, check the PROJECT_DETAILS.md for algorithm explanations and architecture diagrams.
