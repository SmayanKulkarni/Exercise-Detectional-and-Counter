import os
import cv2
import mediapipe as mp
import numpy as np

# --- Configuration ---
DATA_DIR = os.path.expanduser('~/Downloads/Exercises/Data')
OUTPUT_DIR = os.path.expanduser('~/Downloads/Exercises/processed_data')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.webm') 
VISIBILITY_THRESHOLD = 0.5
NUM_BODY_LANDMARKS = 22  # <-- CHANGED: 33 total - 11 face = 22

# --- Sliding Window Parameters ---
WINDOW_SIZE = 60  # Frames per sample
STEP_SIZE = 20    # Overlap

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# --- Automatically detect exercise folders (including 'idle') ---
try:
    exercise_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    if not exercise_folders:
        print(f"Error: No subdirectories found in {DATA_DIR}")
        exit()
    if 'idle' not in exercise_folders:
        print("Warning: 'idle' class folder not found. Real-time stability will be poor.")
    print(f"Found exercise classes: {exercise_folders}")
except FileNotFoundError:
    print(f"Error: Data directory not found at {DATA_DIR}")
    exit()


# Create the output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for exercise in exercise_folders:
    os.makedirs(os.path.join(OUTPUT_DIR, exercise), exist_ok=True)

# --- Main Processing Loop ---
total_samples_created = 0
for exercise in exercise_folders:
    exercise_path = os.path.join(DATA_DIR, exercise)
    output_exercise_path = os.path.join(OUTPUT_DIR, exercise)
    
    video_files = [f for f in os.listdir(exercise_path) if f.lower().endswith(VIDEO_EXTENSIONS)]
    print(f"\nProcessing {exercise} ({len(video_files)} videos)...")

    for video_file in video_files:
        video_path = os.path.join(exercise_path, video_file)
        base_filename = os.path.splitext(video_file)[0]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error: Could not open video {video_file}")
            continue

        all_video_landmarks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks_list = results.pose_landmarks.landmark
                
                lm_left_hip = landmarks_list[mp_pose.PoseLandmark.LEFT_HIP.value]
                lm_right_hip = landmarks_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
                lm_left_shoulder = landmarks_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                lm_right_shoulder = landmarks_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                hip_visibility_ok = (lm_left_hip.visibility > VISIBILITY_THRESHOLD and 
                                     lm_right_hip.visibility > VISIBILITY_THRESHOLD)
                
                shoulder_visibility_ok = (lm_left_shoulder.visibility > VISIBILITY_THRESHOLD and 
                                          lm_right_shoulder.visibility > VISIBILITY_THRESHOLD)
                
                # Still extract all 33 for normalization calculations
                frame_landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])

                if hip_visibility_ok:
                    left_hip_coords = frame_landmarks_np[mp_pose.PoseLandmark.LEFT_HIP.value]
                    right_hip_coords = frame_landmarks_np[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    center_point = (left_hip_coords + right_hip_coords) / 2.0
                    normalized_landmarks = frame_landmarks_np - center_point
                    # <-- CHANGED: Slice off face landmarks (0-10)
                    all_video_landmarks.append(normalized_landmarks[11:]) 
                
                elif shoulder_visibility_ok:
                    left_shoulder_coords = frame_landmarks_np[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder_coords = frame_landmarks_np[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    center_point = (left_shoulder_coords + right_shoulder_coords) / 2.0
                    normalized_landmarks = frame_landmarks_np - center_point
                    # <-- CHANGED: Slice off face landmarks (0-10)
                    all_video_landmarks.append(normalized_landmarks[11:])
                
                else:
                    # <-- CHANGED: Append zeros with the new shape
                    all_video_landmarks.append(np.zeros((NUM_BODY_LANDMARKS, 3)))

            else:
                # <-- CHANGED: Append zeros with the new shape
                all_video_landmarks.append(np.zeros((NUM_BODY_LANDMARKS, 3)))
        
        cap.release()

        if not all_video_landmarks:
            print(f"  No valid landmarks detected in {video_file}")
            continue

        all_video_landmarks_np = np.array(all_video_landmarks)
        total_frames = all_video_landmarks_np.shape[0]
        sample_index = 0

        if total_frames < WINDOW_SIZE:
            # <-- CHANGED: Pad with the new shape
            padded_data = np.zeros((WINDOW_SIZE, NUM_BODY_LANDMARKS, 3))
            padded_data[0:total_frames, :, :] = all_video_landmarks_np
            
            output_npy_path = os.path.join(output_exercise_path, f"{base_filename}_sample_{sample_index}.npy")
            np.save(output_npy_path, padded_data)
            print(f"  Saved {video_file} (padded) -> 1 sample")
            total_samples_created += 1
        else:
            for start_frame in range(0, total_frames - WINDOW_SIZE + 1, STEP_SIZE):
                end_frame = start_frame + WINDOW_SIZE
                window = all_video_landmarks_np[start_frame:end_frame]
                
                output_npy_path = os.path.join(output_exercise_path, f"{base_filename}_sample_{sample_index}.npy")
                np.save(output_npy_path, window)
                sample_index += 1
            
            print(f"  Saved {video_file} -> {sample_index} samples")
            total_samples_created += sample_index

pose.close()
print(f"\n--- Preprocessing Complete ---")
print(f"Total .npy samples created: {total_samples_created}") 