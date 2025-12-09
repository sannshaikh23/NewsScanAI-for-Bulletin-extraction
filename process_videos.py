import os
import subprocess
import sys
from db_utils import get_all_videos, update_video_status

# Force UTF-8 for stdout to handle Hindi filenames
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
WORK_DIR = "work_dir"
KEYFRAMES_DIR = os.path.join(WORK_DIR, "keyframes")
FEATURES_DIR = os.path.join(WORK_DIR, "features")
GMM_MODEL = os.path.join(WORK_DIR, "gmm.pickle")
INDEX_FILE = os.path.join(WORK_DIR, "index.npy")

# Scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_KEYFRAMES = os.path.join(BASE_DIR, "indexer", "keyframes", "extract_keyframes.py")
SCRIPT_SIFT = os.path.join(BASE_DIR, "indexer", "local_descriptors", "extract_sift.py")
SCRIPT_INDEX = os.path.join(BASE_DIR, "indexer", "global_descriptors", "index_dataset.py")

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def process_pending_videos():
    videos = get_all_videos()
    pending_videos = [v for v in videos if not v['is_indexed']]
    
    if not pending_videos:
        print("No pending videos to process.")
        return

    print(f"Found {len(pending_videos)} pending videos.")
    
    # Ensure directories exist
    if not os.path.exists(KEYFRAMES_DIR):
        os.makedirs(KEYFRAMES_DIR)
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    videos_to_update = []

    for video in pending_videos:
        video_name = video['name']
        video_path = video['path'] # Relative path, e.g., static/videos/vid.mp4
        
        print(f"\n--- Processing {video_name} ---")
        
        # 1. Extract Keyframes
        # Output: work_dir/keyframes/video_name_keyframes
        kf_out_dir = os.path.join(KEYFRAMES_DIR, f"{video_name}_keyframes")
        
        try:
            run_command(["python", SCRIPT_KEYFRAMES, video_path, kf_out_dir, "1"])
        except subprocess.CalledProcessError:
            print(f"Failed to extract keyframes for {video_name}. Skipping.")
            continue

        # 2. Extract SIFT
        # Output: work_dir/features/video_name_keyframes (to keep structure)
        # We want the SIFT script to output files named like "video_name_keyframes_000001.npy"
        # The SIFT script's main block handles directory recursion and naming if we pass the input dir.
        # It calculates relative path.
        # If we pass input=work_dir/keyframes and output=work_dir/features, 
        # it will find video_name_keyframes/000001.jpg
        # rel_path = video_name_keyframes/000001.jpg
        # name = video_name_keyframes_000001
        # output = work_dir/features/video_name_keyframes_000001.npy
        # This flattens the structure in features dir, which is what we want for the indexer?
        # Let's check indexer.
        # Indexer looks for **/*.npy.
        # So structure doesn't matter too much, but unique names do.
        # The SIFT script naming convention ensures uniqueness if folder names are unique.
        
        # We can run SIFT extraction on the specific keyframe folder to save time, 
        # instead of the whole keyframes root.
        # But we need to be careful about the output path to ensure the naming convention holds.
        # If we run on `work_dir/keyframes/video_name_keyframes`, 
        # rel_path will be `000001.jpg`.
        # name will be `000001`.
        # output will be `output_dir/000001.npy`.
        # THIS IS BAD. We lose the video name prefix.
        
        # So we MUST run it on `work_dir/keyframes` but restrict it to the specific folder?
        # The SIFT script doesn't have a filter.
        # OR we pass the output directory as `work_dir/features` and input as `work_dir/keyframes/video_name_keyframes`
        # AND we modify the SIFT script or rely on it?
        # Wait, if I run on `work_dir/keyframes/video_name_keyframes`, I get `000001.npy`.
        # I need `video_name_keyframes_000001.npy`.
        
        # Workaround:
        # The SIFT script logic:
        # name = os.path.splitext(rel_path)[0].replace(os.sep, '_')...
        
        # If I want the prefix, I should probably run it on `work_dir/keyframes` 
        # BUT that would re-process ALL videos every time. That's slow.
        
        # Better approach:
        # Use the SIFT script on the specific folder, but ensure the output filename includes the video name.
        # The current SIFT script doesn't support adding a prefix.
        # I should probably modify `extract_sift.py` to accept a prefix OR just handle SIFT extraction here in python 
        # by importing the function, but the script handles the looping and saving nicely.
        
        # Let's look at `extract_sift.py` again.
        # It takes input directory.
        
        # I will modify `extract_sift.py` to be more flexible? 
        # Or I can just create a temporary symlink? No, Windows.
        
        # Let's just use the existing script but be clever.
        # If I run it on `work_dir/keyframes`, it processes everything.
        # If I want to process just one, I need to modify the script or write my own loop here.
        # Writing my own loop here using `extract_sift` function from the module is cleaner.
        
        # Let's import extract_sift from the script.
        # But I need to replicate the naming logic.
        
        from indexer.local_descriptors.extract_sift import extract_sift
        import glob
        
        kf_files = glob.glob(os.path.join(kf_out_dir, "*.jpg"))
        print(f"Extracting features for {len(kf_files)} frames...")
        
        for kf_file in kf_files:
            # Construct desired output filename
            # kf_file = .../test_video_0_keyframes/000001.jpg
            # desired = .../features/test_video_0_keyframes_000001.npy
            
            basename = os.path.basename(kf_file) # 000001.jpg
            name_no_ext = os.path.splitext(basename)[0] # 000001
            
            # The video folder name is video_name_keyframes
            folder_name = os.path.basename(kf_out_dir)
            
            unique_name = f"{folder_name}_{name_no_ext}" # test_video_0_keyframes_000001
            
            out_file = os.path.join(FEATURES_DIR, unique_name + ".npy")
            
            if not os.path.exists(out_file):
                extract_sift(kf_file, out_file)
        
        videos_to_update.append(video_name)

    if not videos_to_update:
        print("No videos successfully processed.")
        return

    # 3. Re-Index Dataset
    # This processes ALL features in FEATURES_DIR.
    print("\n--- Updating Index ---")
    try:
        run_command(["python", SCRIPT_INDEX, FEATURES_DIR, GMM_MODEL, INDEX_FILE])
    except subprocess.CalledProcessError:
        print("Failed to update index.")
        return

    # 4. Update Database Status
    print("\n--- Updating Database Status ---")
    for video_name in videos_to_update:
        update_video_status(video_name, True)
        print(f"Marked {video_name} as indexed.")

if __name__ == "__main__":
    process_pending_videos()
