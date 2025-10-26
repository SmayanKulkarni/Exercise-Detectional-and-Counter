import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from collections import deque

# --- Helper: Calculate Angle ---
# (Unchanged)
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba); mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0: return 90.0
    cosine_angle = np.clip(dot_product / (mag_ba * mag_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# --- "No-Bounce" Rep Counter (For Squat, Curls, Pushup) ---
# (Unchanged)
class RepCounter:
    def __init__(self, down_threshold, up_threshold, exercise_name="exercise", **kwargs): # Added exercise_name for debug
        self.count = 0
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = 'up'
        self.was_deep_enough = False
        self.exercise_name = exercise_name # For printing

    def update(self, angle):
        rep_status = None
        # --- State 'up' -> 'down' transition ---
        if self.state == 'up' and angle < self.up_threshold:
            # print(f"DEBUG [{self.exercise_name}]: State changing UP -> DOWN (Angle: {angle:.1f} < {self.up_threshold})") # DEBUG
            self.state = 'down'
            self.was_deep_enough = False # Reset depth flag
            rep_status = "down"

        # --- State 'down' -> 'up' transition ---
        elif self.state == 'down' and angle > self.up_threshold:
            # print(f"DEBUG [{self.exercise_name}]: Reached UP threshold (Angle: {angle:.1f} > {self.up_threshold})") # DEBUG
            # Check if they went deep enough *before* resetting state/flag
            if self.was_deep_enough:
                self.count += 1
                # print(f"DEBUG [{self.exercise_name}]: Rep Counted! Total: {self.count} (Was deep enough)") # DEBUG
                rep_status = "rep_counted"
            else:
                # print(f"DEBUG [{self.exercise_name}]: Rep NOT Counted (Was NOT deep enough)") # DEBUG
                rep_status = "partial_rep"

            # Now reset state and flag for the next rep
            self.state = 'up'
            self.was_deep_enough = False

        # --- While 'down', check if they hit the 'deep enough' threshold ---
        if self.state == 'down' and angle < self.down_threshold:
            # if not self.was_deep_enough: # Only print the first time it becomes true
            #      print(f"DEBUG [{self.exercise_name}]: Reached DOWN threshold (Angle: {angle:.1f} < {self.down_threshold})") # DEBUG
            self.was_deep_enough = True # Set the flag

        return rep_status


# --- Inverted Rep Counter (For Lateral Raise) ---
# (Unchanged)
class RepCounterInverted:
    def __init__(self, down_threshold, up_threshold, exercise_name="exercise", **kwargs): # Added exercise_name
        self.count = 0
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = 'down'
        self.was_high_enough = False
        self.exercise_name = exercise_name # For printing

    def update(self, angle):
        rep_status = None
        # State 'down' -> 'up'
        if self.state == 'down' and angle > self.down_threshold:
            # print(f"DEBUG [{self.exercise_name}]: State changing DOWN -> UP (Angle: {angle:.1f} > {self.down_threshold})") # DEBUG
            self.state = 'up'
            self.was_high_enough = False # Reset height flag
            rep_status = "up"
        # State 'up' -> 'down'
        elif self.state == 'up' and angle < self.down_threshold:
            # print(f"DEBUG [{self.exercise_name}]: Reached DOWN threshold (Angle: {angle:.1f} < {self.down_threshold})") # DEBUG
             # Check if they went high enough *before* resetting state/flag
            if self.was_high_enough:
                self.count += 1
                # print(f"DEBUG [{self.exercise_name}]: Rep Counted! Total: {self.count} (Was high enough)") # DEBUG
                rep_status = "rep_counted"
            else:
                # print(f"DEBUG [{self.exercise_name}]: Rep NOT Counted (Was NOT high enough)") # DEBUG
                rep_status = "partial_rep"

            # Now reset state and flag for the next rep
            self.state = 'down'
            self.was_high_enough = False

        # --- While 'up', check if they hit the 'high enough' threshold ---
        if self.state == 'up' and angle > self.up_threshold:
            # if not self.was_high_enough: # Only print the first time
            #      print(f"DEBUG [{self.exercise_name}]: Reached UP threshold (Angle: {angle:.1f} > {self.up_threshold})") # DEBUG
            self.was_high_enough = True # Set the flag

        return rep_status
# --- End of Counter Classes ---


# --- Lenient Form Correction ---
# (Unchanged)
def check_form(exercise, landmarks, rep_counter):
    try:
        if exercise == 'squat' and rep_counter:
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            torso_angle = calculate_angle(shoulder, hip, knee)

            if rep_counter.state == 'down' and torso_angle < 60:
                return "Back not straight"

        elif exercise == 'plank':
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            body_angle = calculate_angle(shoulder, hip, ankle)

            if body_angle < 155: return "Hips sagging"
            if body_angle > 175: return "Hips too high"
    except Exception as e:
        pass

    return None
# --- End Form Correction ---


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
DEFAULT_CONSENSUS_COUNT = 1
FAST_CHANGE_COUNT = 2
SLOW_TO_IDLE_COUNT = 13
# --- END OF CORRECTED CONSTANTS ---

prediction_history = deque(maxlen=HISTORY_LEN)
stable_prediction = "idle"

# --- MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Real-Time Variables ---
cap = cv2.VideoCapture("/home/smayan/Downloads/Exercises/Testing Videos/pushup.webm")
# cap = cv2.VideoCapture(0)

landmark_sequence = []
last_prediction_time = time.time()
form_feedback = ""
feedback_display_time = 0

# --- Rep Counter Dictionaries ---
# --- UPDATED: Hammer Curl down_threshold changed to 95 ---
rep_counter_config = {
    'squat': {'class': RepCounter, 'params': {'down_threshold': 90, 'up_threshold': 160, 'exercise_name': 'squat'}},
    'push-up': {'class': RepCounter, 'params': {'down_threshold': 90, 'up_threshold': 160, 'exercise_name': 'push-up'}},
    'barbell biceps curl': {'class': RepCounter, 'params': {'down_threshold': 60, 'up_threshold': 150, 'exercise_name': 'bicep curl'}},
    'hammer curl': {'class': RepCounter, 'params': {'down_threshold': 95, 'up_threshold': 140, 'exercise_name': 'hammer curl'}}, # <-- Adjusted down_threshold!
    'lateral raise': {'class': RepCounterInverted, 'params': {'down_threshold': 30, 'up_threshold': 80, 'exercise_name': 'lateral raise'}},
}
current_rep_counter = None
plank_timer_start = None
plank_grace_period_start = None

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
                        if majority_prediction != 'plank':
                            plank_timer_start = None
                            plank_grace_period_start = None
                        elif majority_prediction == 'plank' and stable_prediction != 'plank':
                             plank_grace_period_start = time.time()
                             plank_timer_start = None

                        stable_prediction = majority_prediction
                        form_feedback = ""

                        if stable_prediction in rep_counter_config:
                            config = rep_counter_config[stable_prediction]
                            current_rep_counter = config['class'](**config['params'])
                        else:
                            current_rep_counter = None

            except ValueError:
                pass

    # --- Exercise-Specific Logic (Counter / Timer / Form) ---
    display_text = f"{stable_prediction}"

    if stable_prediction == 'plank':
        if plank_grace_period_start is not None:
            if plank_timer_start is None:
                grace_time_elapsed = time.time() - plank_grace_period_start
                if grace_time_elapsed >= 5.0:
                    plank_timer_start = time.time()
                    plank_time_elapsed = 0.0
                    cv2.putText(frame, f"Timer: {plank_time_elapsed:.1f}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    countdown = 5.0 - grace_time_elapsed
                    cv2.putText(frame, f"Starting in: {countdown:.0f}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                plank_time_elapsed = time.time() - plank_timer_start
                cv2.putText(frame, f"Timer: {plank_time_elapsed:.1f}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if results.pose_landmarks:
            feedback = check_form('plank', results.pose_landmarks.landmark, None)
            if feedback:
                form_feedback = feedback
                feedback_display_time = time.time()

    elif stable_prediction in rep_counter_config and current_rep_counter:
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            angle = 0 # Initialize angle to 0

            try:
                # --- Ambidextrous Angle Calculation ---
                if stable_prediction == 'squat':
                    shoulder, hip, knee = (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE)
                    left_vis = lm[hip.value].visibility
                    right_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
                    chosen_hip = hip
                    if right_vis > left_vis and right_vis > VISIBILITY_THRESHOLD:
                        shoulder, hip, knee = (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE)
                        chosen_hip = hip
                    if lm[chosen_hip.value].visibility > VISIBILITY_THRESHOLD:
                        angle = calculate_angle([lm[shoulder.value].x, lm[shoulder.value].y, lm[shoulder.value].z],[lm[hip.value].x, lm[hip.value].y, lm[hip.value].z],[lm[knee.value].x, lm[knee.value].y, lm[knee.value].z])

                elif stable_prediction in ['barbell biceps curl', 'hammer curl', 'push-up']:
                    shoulder, elbow, wrist = (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)
                    left_vis = lm[elbow.value].visibility
                    right_vis = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
                    chosen_elbow = elbow
                    if right_vis > left_vis and right_vis > VISIBILITY_THRESHOLD:
                        shoulder, elbow, wrist = (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
                        chosen_elbow = elbow

                    elbow_visibility = lm[chosen_elbow.value].visibility
                    # if stable_prediction == 'hammer curl':
                    #    print(f"Chosen elbow: {chosen_elbow.name}, Visibility: {elbow_visibility:.2f}")

                    if elbow_visibility > VISIBILITY_THRESHOLD:
                        angle = calculate_angle([lm[shoulder.value].x, lm[shoulder.value].y, lm[shoulder.value].z],[lm[elbow.value].x, lm[elbow.value].y, lm[elbow.value].z],[lm[wrist.value].x, lm[wrist.value].y, lm[wrist.value].z])

                elif stable_prediction == 'lateral raise':
                    hip, shoulder, elbow = (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW)
                    left_vis = lm[shoulder.value].visibility
                    right_vis = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                    chosen_shoulder = shoulder
                    if right_vis > left_vis and right_vis > VISIBILITY_THRESHOLD:
                        hip, shoulder, elbow = (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW)
                        chosen_shoulder = shoulder
                    if lm[chosen_shoulder.value].visibility > VISIBILITY_THRESHOLD:
                        angle = calculate_angle([lm[hip.value].x, lm[hip.value].y, lm[hip.value].z],[lm[shoulder.value].x, lm[shoulder.value].y, lm[shoulder.value].z],[lm[elbow.value].x, lm[elbow.value].y, lm[elbow.value].z])
                # --- End of Update ---

                # --- Debugging print for hammer curl ---
                # if stable_prediction == 'hammer curl':
                #    print(f"Hammer Curl Angle: {angle:.1f}, State: {current_rep_counter.state}, Deep Enough: {current_rep_counter.was_deep_enough}")
                # --- End Debugging ---

                if angle > 0:
                    rep_status = current_rep_counter.update(angle)
                    if rep_status == "rep_counted":
                         print(f"{stable_prediction.replace('_', ' ').title()} Rep Counted! Total: {current_rep_counter.count}")
                    elif rep_status == "partial_rep":
                        feedback_text = "Go deeper"
                        if stable_prediction == 'lateral raise':
                            feedback_text = "Raise higher"
                        form_feedback = feedback_text
                        feedback_display_time = time.time()

                feedback = check_form(stable_prediction, lm, current_rep_counter)
                if feedback:
                    form_feedback = feedback
                    feedback_display_time = time.time()

            except Exception as e:
                print(f"Error calculating angle or updating counter: {e}")
                pass

        cv2.putText(frame, f"Reps: {current_rep_counter.count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    elif stable_prediction == 'idle':
        pass

    # --- Display Information ---
    cv2.putText(frame, display_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if form_feedback and (time.time() - feedback_display_time < 2.0): # Show for 2 seconds
        cv2.putText(frame, form_feedback, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        form_feedback = ""

    cv2.imshow('Exercise Classifier', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()