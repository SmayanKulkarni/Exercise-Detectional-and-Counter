import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from collections import deque

# --- Helper: Calculate Angle ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba); mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0: return 90.0
    cosine_angle = np.clip(dot_product / (mag_ba * mag_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# --- UPDATED: "No-Bounce" Rep Counter ---
class RepCounter:
    def __init__(self, down_threshold, up_threshold, **kwargs):
        self.count = 0
        self.down_threshold = down_threshold  # e.g., 90°
        self.up_threshold = up_threshold    # e.g., 160°
        self.state = 'up'
        self.was_deep_enough = False # Flag to track if they hit the bottom
        
    def update(self, angle):
        # --- State 'up' -> 'down' transition ---
        # User is 'up' and has gone *below* the up_threshold (started rep)
        if self.state == 'up' and angle < self.up_threshold:
            self.state = 'down'
            self.was_deep_enough = False # Reset depth flag for this new rep

        # --- State 'down' -> 'up' transition ---
        # User is 'down' and has gone *above* the up_threshold (finished rep)
        elif self.state == 'down' and angle > self.up_threshold:
            # Check if they went deep enough *before* counting
            if self.was_deep_enough:
                self.count += 1
                print(f"Rep Counted! Total: {self.count}")
            
            # Reset state and flag regardless
            self.state = 'up'
            self.was_deep_enough = False

        # --- While 'down', check if they hit the 'deep enough' threshold ---
        if self.state == 'down' and angle < self.down_threshold:
            self.was_deep_enough = True # Set the flag
# --- End of Updated Rep Counter ---


# --- Load Model and Encoder ---
print("Loading model and label encoder...")
model = load_model('exercise_classifier_conv1d_lstm.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
print("Load complete.")

# --- USER-DEFINED CONSTANTS ---
SEQ_LENGTH = 60
NUM_BODY_LANDMARKS = 22
NUM_FEATURES = NUM_BODY_LANDMARKS * 3
PREDICTION_THRESHOLD = 0.5
PREDICTION_INTERVAL = 0.2
VISIBILITY_THRESHOLD = 0.5

# --- CORRECTED Asymmetric Stability ---
HISTORY_LEN = 15 
DEFAULT_CONSENSUS_COUNT = 4   # (Idle -> Exercise) - FASTER LOCK ON
FAST_CHANGE_COUNT = 4         # (Exercise -> Exercise)
SLOW_TO_IDLE_COUNT = 5       # (Exercise -> Idle) - SLOWER LOCK OFF
# --- END OF CORRECTED CONSTANTS ---

prediction_history = deque(maxlen=HISTORY_LEN)
stable_prediction = "idle"

# --- MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Real-Time Variables ---
# --- USER-DEFINED VIDEO PATH ---
cap = cv2.VideoCapture("/home/smayan/Downloads/Exercises/VID_20251026_175453486.mp4")
# cap = cv2.VideoCapture(0) # <-- Uncomment this for webcam

landmark_sequence = []
last_prediction_time = time.time()

# --- Rep Counter Dictionaries ---
# The down_threshold is now used again
rep_counter_config = {
    'squat': RepCounter(down_threshold=90, up_threshold=160),
    'push-up': RepCounter(down_threshold=90, up_threshold=160),
    'barbell biceps curl': RepCounter(down_threshold=60, up_threshold=150),
    'hammer curl': RepCounter(down_threshold=60, up_threshold=150),
    'lateral raise': RepCounter(down_threshold=80, up_threshold=110),
}
current_rep_counter = None
plank_timer_start = None

print("\n--- Starting Real-Time Inference ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("Video finished or failed to read frame.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    current_landmarks_flat = np.zeros(NUM_FEATURES) 

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks_list = results.pose_landmarks.landmark
        frame_landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])

        lm_left_hip = landmarks_list[mp_pose.PoseLandmark.LEFT_HIP.value]
        lm_right_hip = landmarks_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lm_left_shoulder = landmarks_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        lm_right_shoulder = landmarks_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        hip_visibility_ok = (lm_left_hip.visibility > VISIBILITY_THRESHOLD and 
                             lm_right_hip.visibility > VISIBILITY_THRESHOLD)
        shoulder_visibility_ok = (lm_left_shoulder.visibility > VISIBILITY_THRESHOLD and 
                                  lm_right_shoulder.visibility > VISIBILITY_THRESHOLD)
        
        if hip_visibility_ok:
            center_point = (frame_landmarks_np[mp_pose.PoseLandmark.LEFT_HIP.value] + 
                            frame_landmarks_np[mp_pose.PoseLandmark.RIGHT_HIP.value]) / 2.0
            normalized_landmarks = frame_landmarks_np - center_point
            current_landmarks_flat = normalized_landmarks[11:].flatten()
        elif shoulder_visibility_ok:
            center_point = (frame_landmarks_np[mp_pose.PoseLandmark.LEFT_SHOULDER.value] + 
                            frame_landmarks_np[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]) / 2.0
            normalized_landmarks = frame_landmarks_np - center_point
            current_landmarks_flat = normalized_landmarks[11:].flatten()
    
    landmark_sequence.append(current_landmarks_flat)
    landmark_sequence = landmark_sequence[-SEQ_LENGTH:] 

    # --- Classification & Stability Logic ---
    if (time.time() - last_prediction_time > PREDICTION_INTERVAL) and len(landmark_sequence) == SEQ_LENGTH:
        last_prediction_time = time.time()
        
        padded_sequence = pad_sequences([landmark_sequence], maxlen=SEQ_LENGTH, dtype='float32', padding='pre')
        padded_sequence = padded_sequence.reshape(1, SEQ_LENGTH, NUM_FEATURES) 

        prediction = model.predict(padded_sequence, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        prediction_probability = prediction[predicted_class_index]
        
        current_prediction = "idle"
        if prediction_probability > PREDICTION_THRESHOLD:
            current_prediction = le.classes_[predicted_class_index]
        
        prediction_history.append(current_prediction)
        
        if len(prediction_history) == HISTORY_LEN:
            try:
                majority_prediction = max(set(prediction_history), key=list(prediction_history).count)
                consensus_count = list(prediction_history).count(majority_prediction)
                
                if majority_prediction != stable_prediction:
                    
                    required_count = DEFAULT_CONSENSUS_COUNT
                    if stable_prediction != 'idle' and majority_prediction != 'idle':
                        required_count = FAST_CHANGE_COUNT
                    elif stable_prediction != 'idle' and majority_prediction == 'idle':
                        required_count = SLOW_TO_IDLE_COUNT
                    
                    if consensus_count >= required_count:
                        print(f"--- STATE CHANGE: Locked in '{majority_prediction}' (Count: {consensus_count}/{required_count}) ---")
                        stable_prediction = majority_prediction
                        
                        plank_timer_start = None
                        if stable_prediction in rep_counter_config:
                            if current_rep_counter is None or current_rep_counter.__class__ != rep_counter_config[stable_prediction].__class__:
                                current_rep_counter = rep_counter_config[stable_prediction]
                                current_rep_counter.count = 0 
                        else:
                            current_rep_counter = None
                            
            except ValueError: 
                pass 

    # --- Exercise-Specific Logic (Counter / Timer) ---
    display_text = f"{stable_prediction}"
    
    if stable_prediction == 'plank':
        if plank_timer_start is None:
            plank_timer_start = time.time()
        plank_time_elapsed = time.time() - plank_timer_start
        cv2.putText(frame, f"Timer: {plank_time_elapsed:.1f}s", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    elif stable_prediction in rep_counter_config and current_rep_counter:
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            angle = 0
            
            try:
                if stable_prediction == 'squat':
                    angle = calculate_angle(
                        [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y, lm[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                        [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z])
                
                elif stable_prediction in ['barbell biceps curl', 'hammer curl']:
                    angle = calculate_angle(
                        [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
                        [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].z])
                
                # (Add angles for push-up and lateral raise)
                
                # --- Optional Debugging ---
                # if stable_prediction == 'squat':
                #    print(f"Angle: {angle:.2f}, State: {current_rep_counter.state}, Deep: {current_rep_counter.was_deep_enough}")
                # --- End Debugging ---

                if angle > 0:
                    current_rep_counter.update(angle)

            except Exception as e:
                print(f"Error calculating angle: {e}")
                pass 

        cv2.putText(frame, f"Reps: {current_rep_counter.count}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    elif stable_prediction == 'idle':
        plank_timer_start = None

    # --- Display Information ---
    cv2.putText(frame, display_text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Exercise Classifier', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()