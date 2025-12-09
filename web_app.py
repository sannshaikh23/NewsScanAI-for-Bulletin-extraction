from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import cv2
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables for loaded models
index = None
gmm = None
index_path = "work_dir/index.npy"
gmm_path = "work_dir/gmm.pickle"
keyframes_dir = "work_dir/keyframes"

def load_models():
    global index, gmm
    if index is None:
        print("Loading index...")
        index = np.load(index_path, allow_pickle=True).item()
    if gmm is None:
        print("Loading GMM...")
        with open(gmm_path, 'rb') as f:
            gmm = pickle.load(f)

def compute_fisher_vector(descriptors, gmm):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(gmm.n_components * 2 * 128, dtype=np.float32)
    q = gmm.predict_proba(descriptors)
    D = descriptors.shape[1]
    means = gmm.means_
    covars = gmm.covariances_
    sigma = np.sqrt(covars)
    inv_sigma = 1.0 / sigma
    weights = gmm.weights_
    Q = np.sum(q, axis=0)
    S = np.dot(q.T, descriptors)
    S_sq = np.dot(q.T, descriptors ** 2)
    G_mu = (S - Q[:, np.newaxis] * means) * inv_sigma
    G_mu = G_mu / np.sqrt(weights[:, np.newaxis])
    G_sigma = (S_sq - 2 * means * S + Q[:, np.newaxis] * (means ** 2))
    G_sigma = G_sigma * (inv_sigma ** 2) - Q[:, np.newaxis]
    G_sigma = G_sigma / np.sqrt(2 * weights[:, np.newaxis])
    fv = np.concatenate([G_mu.flatten(), G_sigma.flatten()])
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    norm = np.linalg.norm(fv)
    if norm > 0:
        fv = fv / norm
    return fv

def extract_sift_features(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Resize if image is too large to prevent memory issues
        max_dim = 1024
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors = descriptors.astype(np.float32)
        return descriptors
    except Exception as e:
        print(f"Error extracting SIFT features: {e}")
        return None

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    load_models()
    
    # Track if this is an uploaded image vs sample image
    is_uploaded_image = False
    
    if 'query_image' in request.files:
        # Handle uploaded image
        file = request.files['query_image']
        temp_path = "temp_query.jpg"
        file.save(temp_path)
        query_path = temp_path
        is_uploaded_image = True
        print(f"Uploaded image saved to {temp_path}")
    elif 'query_path' in request.form:
        # Handle path to existing image (sample)
        query_path = request.form['query_path']
        is_uploaded_image = False
        print(f"Using existing image: {query_path}")
        if not os.path.exists(query_path):
            print(f"ERROR: Image path does not exist: {query_path}")
            return jsonify({'error': f'Image not found: {query_path}'}), 400
    else:
        print("ERROR: No image provided in request")
        return jsonify({'error': 'No image provided'}), 400
    
    # Extract features and compute FV
    descriptors = extract_sift_features(query_path)
    if descriptors is None:
        print(f"ERROR: Could not extract features from {query_path}")
        return jsonify({'error': 'Could not extract features'}), 400
    
    print(f"Extracted {len(descriptors)} SIFT descriptors")
    query_fv = compute_fisher_vector(descriptors, gmm)
    
    # Search
    scores = []
    for name, fv in index.items():
        score = np.dot(query_fv, fv)
        scores.append((name, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Set minimum score threshold to filter out poor matches
    # Use different thresholds for uploaded images vs sample images
    if len(scores) > 0:
        top_score = scores[0][1]
        if is_uploaded_image:
            # Strict filter for uploaded images
            # Use BOTH absolute minimum (0.50) AND relative threshold (85% of top score)
            # This prevents showing results when there's no real match
            absolute_min = 0.50
            relative_threshold = top_score * 0.85
            min_score_threshold = max(absolute_min, relative_threshold)
            print(f"Using strict threshold for uploaded image: {min_score_threshold:.4f} (absolute: {absolute_min}, relative: {relative_threshold:.4f})")
        else:
            # Relaxed filter for sample images (50% of top score)
            min_score_threshold = max(0.3, top_score * 0.5)
            print(f"Using relaxed threshold for sample image: {min_score_threshold:.4f}")
    else:
        min_score_threshold = 0.3
    
    # Get top results that meet the threshold (max 10)
    results = []
    for i in range(min(10, len(scores))):
        name, score = scores[i]
        
        # Stop if score drops below threshold
        if score < min_score_threshold:
            break
        
        # Parse the new naming scheme: test_video_X_keyframes_NNNNNN
        video_name = None
        frame_number = None
        img_path = None
        
        # Try to parse the name
        if '_keyframes_' in name:
            parts = name.split('_keyframes_')
            if len(parts) == 2:
                video_name = parts[0]  # e.g., test_video_0
                try:
                    frame_number = int(parts[1])  # e.g., 000001 -> 1
                except:
                    pass
                
                # Construct the image path
                img_path = os.path.join(keyframes_dir, f"{video_name}_keyframes", f"{parts[1]}.jpg")
        
        if img_path and os.path.exists(img_path):
            results.append({
                'name': name,
                'score': float(score),
                'image': image_to_base64(img_path),
                'video_name': video_name,
                'frame_number': frame_number
            })
    
    # Get query image
    query_image_b64 = image_to_base64(query_path)
    
    # If no results meet the threshold, return a helpful message
    if len(results) == 0 and len(scores) > 0:
        return jsonify({
            'query_image': query_image_b64,
            'results': [],
            'message': f'No matching videos found. Best similarity score was {scores[0][1]:.4f}, which is below the threshold of {min_score_threshold:.4f}.'
        })
    
    return jsonify({
        'query_image': query_image_b64,
        'results': results
    })

@app.route('/play_video/<video_name>/<int:frame_number>')
def play_video(video_name, frame_number):
    # Calculate timestamp (assuming 1 fps extraction rate)
    timestamp = frame_number - 1  # Frame 1 is at 0 seconds
    
    # Get video URL from database (Cloudinary or local)
    from db_utils import get_video_url
    video_url = get_video_url(video_name)
    
    return render_template('video_player.html', 
                         video_name=video_name, 
                         timestamp=timestamp,
                         frame_number=frame_number,
                         video_url=video_url)


@app.route('/get_sample_images')
def get_sample_images():
    from db_utils import get_sample_videos
    samples = []
    videos = get_sample_videos()
    
    for video in videos:
        # Construct path to first frame
        # Assuming keyframes are in work_dir/keyframes/<video_name>_keyframes/000001.jpg
        # This naming convention is from the extraction script
        video_name = video['name']
        video_dir = os.path.join(keyframes_dir, f"{video_name}_keyframes")
        
        if os.path.exists(video_dir):
            first_img = os.path.join(video_dir, "000001.jpg")
            if os.path.exists(first_img):
                try:
                    with open(first_img, 'rb') as f:
                        img_data = f.read()
                    if len(img_data) > 0:
                        samples.append({
                            'path': first_img,
                            'name': f"{video_name} - Frame 1",
                            'image': base64.b64encode(img_data).decode('utf-8')
                        })
                except Exception as e:
                    print(f"Error loading sample image {first_img}: {e}")
    return jsonify(samples)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
