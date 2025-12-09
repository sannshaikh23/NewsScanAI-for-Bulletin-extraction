import sqlite3
import os
from datetime import datetime

DB_NAME = "videosearch.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with the videos table."""
    conn = get_db_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_indexed BOOLEAN DEFAULT 0,
                frame_count INTEGER DEFAULT 0,
                show_in_samples BOOLEAN DEFAULT 0
            )
        ''')
    conn.close()
    print(f"Database {DB_NAME} initialized.")

def add_video(name, path, is_indexed=False, frame_count=0, show_in_samples=False):
    """Add a new video to the database."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute('''
                INSERT INTO videos (name, path, is_indexed, frame_count, show_in_samples)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, path, is_indexed, frame_count, show_in_samples))
        print(f"Video {name} added to database.")
        return True
    except sqlite3.IntegrityError:
        print(f"Video {name} already exists.")
        return False
    finally:
        conn.close()

def get_all_videos():
    """Retrieve all videos from the database."""
    conn = get_db_connection()
    videos = conn.execute('SELECT * FROM videos').fetchall()
    conn.close()
    return videos

def get_video_by_name(name):
    """Retrieve a video by its name."""
    conn = get_db_connection()
    video = conn.execute('SELECT * FROM videos WHERE name = ?', (name,)).fetchone()
    conn.close()
    return video

def update_video_status(name, is_indexed):
    """Update the indexed status of a video."""
    conn = get_db_connection()
    with conn:
        conn.execute('UPDATE videos SET is_indexed = ? WHERE name = ?', (is_indexed, name))
    conn.close()

def get_sample_videos():
    """Retrieve videos that should be shown in samples."""
    conn = get_db_connection()
    videos = conn.execute('SELECT * FROM videos WHERE show_in_samples = 1').fetchall()
    conn.close()
    return videos

def get_video_url(video_name):
    """Get video URL - returns Cloudinary URL if available, else local path"""
    conn = get_db_connection()
    video = conn.execute(
        'SELECT cloudinary_url, path FROM videos WHERE name = ?', 
        (video_name,)
    ).fetchone()
    conn.close()
    
    if video:
        # Prefer Cloudinary URL if available
        if video['cloudinary_url']:
            return video['cloudinary_url']
        # Fallback to local path
        return f"/static/videos/{video_name}.mp4"
    
    # Default fallback
    return f"/static/videos/{video_name}.mp4"

if __name__ == "__main__":
    init_db()
