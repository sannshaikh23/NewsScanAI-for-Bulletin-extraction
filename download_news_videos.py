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
    else:
        print("\n✗ Download failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
