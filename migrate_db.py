import sqlite3
from db_utils import DB_NAME

def migrate():
    """Add show_in_samples column to existing database."""
    conn = sqlite3.connect(DB_NAME)
    
    # Check if column exists
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(videos)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'show_in_samples' not in columns:
        print("Adding show_in_samples column...")
        conn.execute('ALTER TABLE videos ADD COLUMN show_in_samples BOOLEAN DEFAULT 0')
        conn.commit()
        print("Column added.")
    else:
        print("Column already exists.")
    
    # Update specific videos to show in samples
    # Only test_video_0, test_video_2, test_video_3 should show
    print("\nUpdating video flags...")
    
    # Set all to hidden first
    conn.execute('UPDATE videos SET show_in_samples = 0')
    
    # Show only the original test videos (0, 2, 3)
    for vid in ['test_video_0', 'test_video_2', 'test_video_3']:
        conn.execute('UPDATE videos SET show_in_samples = 1 WHERE name = ?', (vid,))
        print(f"Set {vid} to show in samples")
    
    conn.commit()
    conn.close()
    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
