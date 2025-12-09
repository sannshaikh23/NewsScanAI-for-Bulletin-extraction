import sqlite3
from db_utils import get_db_connection, get_all_videos

def mark_videos_as_samples():
    """Mark some videos to show as samples in the interface."""
    conn = get_db_connection()
    
    # Get all videos
    videos = get_all_videos()
    
    # Mark the first 5 indexed videos as samples
    sample_count = 0
    for video in videos:
        if video['is_indexed'] and sample_count < 5:
            conn.execute('UPDATE videos SET show_in_samples = 1 WHERE name = ?', (video['name'],))
            print(f"Marked {video['name']} as sample")
            sample_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"\nTotal videos marked as samples: {sample_count}")

if __name__ == "__main__":
    mark_videos_as_samples()
