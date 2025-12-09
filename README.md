# News Scan AI for Bulletin Extraction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent AI-powered system for searching and extracting news bulletin segments from video archives using advanced computer vision techniques.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Demo](#-demo)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

**News Scan AI for Bulletin Extraction** is an intelligent video search system designed specifically for news organizations and media houses. It enables journalists, editors, and researchers to quickly find specific news segments across large video archives by simply uploading a query image or selecting from sample frames.

The system uses state-of-the-art computer vision algorithms (SIFT features, Fisher Vectors, and Gaussian Mixture Models) to identify visually similar scenes and automatically navigate to the exact timestamp in the video.

---

## ✨ Features

- 🔍 **Image-Based Search** - Upload any image to find similar scenes in your video database
- 🎬 **Instant Video Playback** - Click on results to play the video at the exact matching timestamp
- 📊 **Sample Queries** - Pre-loaded sample images for quick testing and demonstration
- 🎯 **Smart Filtering** - Adaptive threshold system (0.50 minimum) to show only relevant results
- ⚠️ **User-Friendly Feedback** - Clear "No Matching Videos Found" message for irrelevant queries
- 💾 **Metadata Management** - SQLite database for efficient video organization
- ⚡ **Fast Retrieval** - Fisher Vector encoding for efficient similarity search
- ☁️ **Cloud Storage Support** - Integrated with Cloudinary for scalable video hosting
- 🎨 **Modern UI** - Clean, responsive web interface with professional styling

---

## 🎥 Demo

### Search Interface
![Search Interface](static/demo_screenshot.png)

### Search Results
Upload a query image or select a sample, and the system returns ranked results with similarity scores:

- **Rank #1** - Highest similarity match
- **Score** - Similarity score (0.0 to 1.0)
- **Video Preview** - Thumbnail of the matching frame
- **Click to Play** - Instant video playback at the exact timestamp

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.7+, Flask |
| **Computer Vision** | OpenCV (SIFT features) |
| **Machine Learning** | scikit-learn (GMM, Fisher Vectors) |
| **Database** | SQLite |
| **Frontend** | HTML5, CSS3, JavaScript (ES6) |
| **Video Processing** | FFmpeg |
| **Cloud Storage** | Cloudinary (optional) |

---

## 📦 Installation

### Prerequisites

- **Python 3.7 or higher**
- **FFmpeg** (for video processing)

#### Install FFmpeg on Windows
```powershell
winget install FFmpeg
```

Or download from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

#### Install FFmpeg on Linux
```bash
sudo apt-get install ffmpeg
```

#### Install FFmpeg on macOS
```bash
brew install ffmpeg
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/news-scan-ai.git
cd news-scan-ai
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- opencv-python
- numpy
- scikit-learn
- Flask
- Pillow
- cloudinary (optional)

---

## 🚀 Quick Start

### Option 1: Using Existing Index

If the project already has indexed videos in `work_dir/`:

```bash
python web_app.py
```

Then open your browser to: **http://127.0.0.1:5000**

### Option 2: First-Time Setup

If starting from scratch with new videos:

```bash
# 1. Add videos to static/videos/ (or configure Cloudinary)

# 2. Register videos in database
python populate_db.py

# 3. Process and index videos
python process_videos.py

# 4. Start the web application
python web_app.py
```

---

## 💡 Usage

### Searching for News Segments

#### Method 1: Upload Query Image
1. Click **"Upload Query Image"**
2. Select an image file (screenshot, photo, or frame)
3. View results ranked by similarity
4. Click any result to play the video at that timestamp

#### Method 2: Use Sample Images
1. Click **"Load Sample Images"**
2. Click on any sample thumbnail
3. View results and play videos

### Adding New Videos

```bash
# Step 1: Add video files to static/videos/
# Example: static/videos/news_bulletin_2024.mp4

# Step 2: Register in database
python populate_db.py

# Step 3: Process and index
python process_videos.py

# Step 4: Restart web app
# Press Ctrl+C to stop, then:
python web_app.py
```

**⏱️ Processing Time:** ~2-5 minutes per video (depending on length)

---

## 📁 Project Structure

```
news-scan-ai/
├── web_app.py              # Main Flask application
├── db_utils.py             # Database helper functions
├── populate_db.py          # Register videos in database
├── process_videos.py       # Automated indexing pipeline
├── download_news_videos.py # Download videos from YouTube
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── videosearch.db          # SQLite database
├── cloudinary_urls.json    # Video URL mappings
├── documents/              # Documentation
│   ├── PROJECT_DETAILS.md  # Technical documentation
│   ├── cloudinary_guide.md # Cloudinary setup guide
│   └── algorithmdetails.md # Algorithm details
├── templates/              # HTML templates
│   ├── index.html          # Main search interface
│   └── video_player.html   # Video playback page
├── static/
│   └── videos/             # Video files (MP4) - optional if using Cloudinary
├── work_dir/               # Generated files
│   ├── keyframes/          # Extracted frames
│   ├── features/           # SIFT descriptors
│   ├── gmm.pickle          # Trained Gaussian Mixture Model
│   └── index.npy           # Search index (Fisher Vectors)
└── indexer/                # Processing scripts
    ├── keyframes/          # Keyframe extraction
    ├── local_descriptors/  # SIFT feature extraction
    └── global_descriptors/ # Fisher Vector encoding
```

---

## 🔬 How It Works

### Algorithm Pipeline

1. **Keyframe Extraction**
   - Extract frames at 1 FPS from videos using FFmpeg
   - Store frames in `work_dir/keyframes/`

2. **Feature Extraction**
   - Extract SIFT (Scale-Invariant Feature Transform) descriptors from each frame
   - SIFT features are robust to scale, rotation, and illumination changes

3. **Fisher Vector Encoding**
   - Train a Gaussian Mixture Model (GMM) with 256 components
   - Encode SIFT descriptors as Fisher Vectors for efficient comparison

4. **Indexing**
   - Build a searchable index of all Fisher Vectors
   - Store in `work_dir/index.npy` for fast retrieval

5. **Query Processing**
   - Extract SIFT features from query image
   - Encode as Fisher Vector
   - Compute cosine similarity with all indexed frames
   - Apply adaptive threshold (0.50 minimum for uploaded images)
   - Return top matches ranked by similarity

### Similarity Threshold

- **Uploaded Images:** Minimum score of **0.50** (strict filtering)
- **Sample Images:** Minimum score of **0.30** (relaxed filtering)
- **No Match:** Displays "⚠️ No Matching Videos Found" message

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for Cloudinary configuration (optional):

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### Adjusting Similarity Threshold

Edit `web_app.py` line 139 to adjust the minimum threshold:

```python
absolute_min = 0.50  # Adjust between 0.30-0.60
```

- **0.30-0.45:** Lenient (may show some irrelevant results)
- **0.45-0.50:** Moderate (recommended)
- **0.50-0.60:** Strict (only very similar images)

---

## 🐛 Troubleshooting

### Videos Not Playing
- Ensure videos are in `static/videos/` or Cloudinary is configured
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

### Low Similarity Scores
- Use higher quality query images
- Ensure query image is from the same video or similar scene
- Adjust threshold in `web_app.py` if needed

---

## 📚 Documentation

For detailed technical documentation, see:
- [DetailsofProject.md](documents/DetailsofProject.md) - Complete project details including architecture, algorithms, cloud integration, and all technical specifications

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with OpenCV and scikit-learn
- Inspired by modern content-based image retrieval systems
- Designed for news organizations and media professionals

---

<div align="center">

**Made with ❤️ for News Organizations**

[⬆ Back to Top](#news-scan-ai-for-bulletin-extraction)

</div>
