import os
import glob
from db_utils import init_db, add_video

VIDEO_DIR = "static/videos"

def populate_db():
    init_db()
    
    # Find all mp4 files in static/videos
    videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")
    
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        # For now, we assume they are indexed if they exist in the static folder 
        # and we can find them in the index (which we can't check easily here without loading the big index).
        # Let's just add them as is_indexed=True for now since they are the test videos.
        
        # Path relative to where the app runs
        rel_path = v.replace("\\", "/")
        
        # Check if it's already in the DB to avoid resetting status (add_video handles unique constraint but we want to be clear)
        # For new videos, we default to is_indexed=False so they get picked up by the processor
        # We also set show_in_samples=False so new videos don't appear in the sample list automatically
        add_video(name, rel_path, is_indexed=False, show_in_samples=False)

if __name__ == "__main__":
    if not os.path.exists(VIDEO_DIR):
        print(f"Directory {VIDEO_DIR} does not exist!")
    else:
        populate_db()
