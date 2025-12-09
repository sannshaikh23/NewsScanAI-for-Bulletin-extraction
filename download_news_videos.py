"""
Script to download 10 videos from Indian news channels (Aaj Tak, ABP News, etc.)
Uses yt-dlp to download videos from YouTube channels.
"""

import os
import subprocess
import sys

# YouTube channel URLs for Indian news channels
CHANNELS = {
    'aajtak': 'https://www.youtube.com/@aajtak',
    'abpnews': 'https://www.youtube.com/@abpnewstv',
    'ndtv': 'https://www.youtube.com/@ndtv',
}

def check_ytdlp_installed():
    """Check if yt-dlp is installed, if not provide installation instructions."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        print("✓ yt-dlp is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ yt-dlp is not installed")
        print("\nTo install yt-dlp, run:")
        print("  pip install yt-dlp")
        return False

def download_videos(channel_name='aajtak', num_videos=10, output_dir='downloaded_videos'):
    """
    Download videos from a specified Indian news channel.
    
    Args:
        channel_name: Name of the channel ('aajtak', 'abpnews', 'ndtv')
        num_videos: Number of videos to download (default: 10)
        output_dir: Directory to save downloaded videos
    """
    
    if not check_ytdlp_installed():
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get channel URL
    if channel_name.lower() not in CHANNELS:
        print(f"Channel '{channel_name}' not found. Available channels: {', '.join(CHANNELS.keys())}")
        return False
    
    channel_url = CHANNELS[channel_name.lower()]
    
    print(f"\n{'='*60}")
    print(f"Downloading {num_videos} videos from {channel_name.upper()}")
    print(f"Channel URL: {channel_url}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # yt-dlp command to download videos
    # Downloads the latest videos, limits to num_videos
    # Format: best quality up to 720p (to save space and time)
    cmd = [
        'yt-dlp',
        '--playlist-end', str(num_videos),  # Limit to num_videos
        '--format', 'bestvideo[height<=720]+bestaudio/best[height<=720]',  # Max 720p
        '--output', os.path.join(output_dir, '%(channel)s_%(title)s_%(id)s.%(ext)s'),
        '--merge-output-format', 'mp4',  # Merge to mp4
        '--no-playlist',  # Don't download playlists, just recent uploads
        '--max-downloads', str(num_videos),  # Maximum downloads
        '--ignore-errors',  # Continue on download errors
        '--no-warnings',  # Suppress warnings
        f'{channel_url}/videos',  # Channel's videos page
    ]
    
    try:
        print("Starting download... This may take a while depending on video sizes.\n")
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*60}")
        print(f"✓ Successfully downloaded videos to '{output_dir}'")
        print(f"{'='*60}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during download: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
        return False

def main():
    """Main function to run the video downloader."""
    print("Indian News Channel Video Downloader")
    print("=" * 60)
    
    # Default settings
    channel = 'aajtak'  # Default to Aaj Tak
    num_videos = 10
    output_dir = 'downloaded_videos'
    
    # You can modify these settings here:
    # channel = 'abpnews'  # Uncomment to use ABP News
    # channel = 'ndtv'     # Uncomment to use NDTV
    # num_videos = 5       # Change number of videos
    
    print(f"\nSettings:")
    print(f"  Channel: {channel.upper()}")
    print(f"  Number of videos: {num_videos}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Download videos
    success = download_videos(channel, num_videos, output_dir)
    
    if success:
        print("\n✓ All done! Videos are ready in the output directory.")
        
        # List downloaded files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
            if files:
                print(f"\nDownloaded {len(files)} video(s):")
                for i, file in enumerate(files, 1):
                    file_path = os.path.join(output_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  {i}. {file} ({size_mb:.2f} MB)")
                
                # Upload to Cloudinary
                print("\n" + "="*60)
                print("Uploading videos to Cloudinary...")
                print("="*60)
                upload_to_cloudinary(output_dir, files)
    else:
        print("\n✗ Download failed. Please check the error messages above.")
        sys.exit(1)

def upload_to_cloudinary(output_dir, files):
    """Upload downloaded videos to Cloudinary and add to database"""
    try:
        import cloudinary
        import cloudinary.uploader
        from dotenv import load_dotenv
        from db_utils import add_video
        import json
        
        # Load environment variables
        load_dotenv()
        
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )
        
        if not all([os.getenv('CLOUDINARY_CLOUD_NAME'), os.getenv('CLOUDINARY_API_KEY'), os.getenv('CLOUDINARY_API_SECRET')]):
            print("\n⚠️  Cloudinary credentials not found. Videos saved locally only.")
            return
        
        successful = 0
        failed = 0
        
        # Load existing URL mapping
        mapping_file = "cloudinary_urls.json"
        url_mapping = {}
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                url_mapping = json.load(f)
        
        for i, file in enumerate(files, 1):
            file_path = os.path.join(output_dir, file)
            video_name = os.path.splitext(file)[0]  # Remove .mp4 extension
            
            print(f"\n[{i}/{len(files)}] Uploading: {video_name}")
            
            try:
                # Upload to Cloudinary
                response = cloudinary.uploader.upload(
                    file_path,
                    resource_type="video",
                    public_id=f"videosearch/videos/{video_name}",
                    folder="videosearch/videos",
                    overwrite=False,
                    invalidate=True
                )
                
                cloudinary_url = response['secure_url']
                url_mapping[video_name] = cloudinary_url
                
                print(f"   ✅ Uploaded to Cloudinary")
                
                # Add to database
                add_video(
                    name=video_name,
                    path=f"static/videos/{file}",  # Keep local path for reference
                    is_indexed=False,
                    frame_count=0,
                    show_in_samples=False
                )
                
                # Update database with Cloudinary URL
                import sqlite3
                conn = sqlite3.connect("videosearch.db")
                conn.execute(
                    'UPDATE videos SET cloudinary_url = ? WHERE name = ?',
                    (cloudinary_url, video_name)
                )
                conn.commit()
                conn.close()
                
                print(f"   ✅ Added to database")
                successful += 1
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                failed += 1
        
        # Save updated mapping
        with open(mapping_file, 'w') as f:
            json.dump(url_mapping, f, indent=2)
        
        print("\n" + "="*60)
        print(f"📊 Upload Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print("="*60)
        
        print("\n💡 Next steps:")
        print("   1. Run: python process_videos.py (to extract features)")
        print("   2. Videos are now in Cloudinary and ready to use!")
        
    except ImportError as e:
        print(f"\n⚠️  Missing dependencies: {e}")
        print("Videos saved locally. Install cloudinary to enable cloud upload:")
        print("  pip install cloudinary python-dotenv")

if __name__ == '__main__':
    main()

