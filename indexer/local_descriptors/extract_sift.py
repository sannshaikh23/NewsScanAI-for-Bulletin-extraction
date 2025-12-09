import cv2
import numpy as np
import sys
import os
import glob
import argparse

# Force UTF-8 for stdout to handle Hindi filenames
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def extract_sift(image_path, output_path, edge_threshold=5, contrast_threshold=0.04):
    """
    Extracts SIFT features from an image and saves them to a .npy file.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the .npy file.
        edge_threshold (float): Edge threshold for SIFT (default 5 to match original C++ code).
        contrast_threshold (float): Contrast threshold for SIFT.
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return

        # Create SIFT detector
        sift = cv2.SIFT_create(contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)

        # Detect and compute
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)
            keypoints_data = np.zeros((0, 4), dtype=np.float32)
        else:
            # Convert keypoints to numpy array (x, y, size, angle)
            keypoints_data = np.zeros((len(keypoints), 4), dtype=np.float32)
            for i, kp in enumerate(keypoints):
                keypoints_data[i] = [kp.pt[0], kp.pt[1], kp.size, kp.angle]

        # Save as dictionary in .npy
        np.save(output_path, {"keypoints": keypoints_data, "descriptors": descriptors})

    except Exception as e:
        print(f"Exception processing {image_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SIFT features")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--ext", default="jpg", help="Image extension to look for if input is directory")
    
    args = parser.parse_args()

    if os.path.isdir(args.input):
        # Process directory
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        files = glob.glob(os.path.join(args.input, "**", f"*.{args.ext}"), recursive=True)
        print(f"Found {len(files)} images in {args.input}")
        
        for f in files:
            # Get relative path from input directory
            rel_path = os.path.relpath(f, args.input)
            # Remove extension and replace path separators with underscores
            # e.g., test_video_0_keyframes/000001.jpg -> test_video_0_keyframes_000001
            name = os.path.splitext(rel_path)[0].replace(os.sep, '_').replace('/', '_').replace('\\', '_')
            
            output_file = os.path.join(args.output, f"{name}.npy")
            
            # Skip if already exists
            if os.path.exists(output_file):
                continue
                
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read {f}")
                continue
                
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) == 0:
                print(f"No features found in {f}")
                continue
            
            # Convert keypoints to serializable format
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    'pt': kp.pt,
                    'size': kp.size,
                    'angle': kp.angle,
                    'response': kp.response,
                    'octave': kp.octave,
                    'class_id': kp.class_id
                })
            
            # Save as dictionary
            data = {
                'keypoints': kp_data,
                'descriptors': descriptors.astype(np.float32)
            }
            
            np.save(output_file, data)
    else:
        # Process single file
        if os.path.isdir(args.output):
             name = os.path.splitext(os.path.basename(args.input))[0]
             out_path = os.path.join(args.output, name + ".npy")
        else:
            out_path = args.output
            # Ensure parent dir exists
            parent = os.path.dirname(out_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent)
                
        extract_sift(args.input, out_path)
