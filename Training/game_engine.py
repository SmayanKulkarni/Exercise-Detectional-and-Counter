import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from collections import deque
import csv 
import os 
from datetime import datetime
import pandas as pd

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

# --- Rep Counter Classes (RepCounter & RepCounterInverted) ---
# (Unchanged)
class RepCounter:
    def __init__(self, down_threshold, up_threshold, exercise_name="exercise", weight=0, **kwargs):
        self.count = 0
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = 'up'
        self.min_angle_in_rep = 180
        self.form_error = None
        self.exercise_name = exercise_name
        self.weight = weight
        self.base_points = 5
        self.penalty_points = -3

    def update(self, angle):
        rep_status = None; points = 0
        if self.state == 'up':
            if angle < self.up_threshold:
                self.state = 'down'; self.min_angle_in_rep = angle; self.form_error = None; rep_status = "down"
        elif self.state == 'down':
            self.min_angle_in_rep = min(self.min_angle_in_rep, angle)
            if angle > self.up_threshold:
                was_deep_enough = self.min_angle_in_rep < self.down_threshold
                if was_deep_enough and self.form_error is None:
                    self.count += 1; points = self.base_points + int(self.weight * 2); rep_status = "rep_counted"
                else:
                    points = self.penalty_points
                    if not was_deep_enough: rep_status = "partial_rep"
                    else: rep_status = f"form_error:{self.form_error}"
                self.state = 'up'; self.min_angle_in_rep = 180; self.form_error = None
        return rep_status, points
    
    def log_form_error(self, error_message):
        if self.state == 'down': self.form_error = error_message

class RepCounterInverted:
    def __init__(self, down_threshold, up_threshold, exercise_name="exercise", weight=0, **kwargs):
        self.count = 0
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = 'down'
        self.max_angle_in_rep = 0
        self.form_error = None
        self.exercise_name = exercise_name
        self.weight = weight
        self.base_points = 5
        self.penalty_points = -3

    def update(self, angle):
        rep_status = None; points = 0
        if self.state == 'down':
            if angle > self.down_threshold:
                self.state = 'up'; self.max_angle_in_rep = angle; self.form_error = None; rep_status = "up"
        elif self.state == 'up':
            self.max_angle_in_rep = max(self.max_angle_in_rep, angle)
            if angle < self.down_threshold:
                was_high_enough = self.max_angle_in_rep > self.up_threshold
                if was_high_enough and self.form_error is None:
                    self.count += 1; points = self.base_points + int(self.weight * 2); rep_status = "rep_counted"
                else:
                    points = self.penalty_points
                    if not was_high_enough: rep_status = "partial_rep"
                    else: rep_status = f"form_error:{self.form_error}"
                self.state = 'down'; self.max_angle_in_rep = 0; self.form_error = None
        return rep_status, points

    def log_form_error(self, error_message):
        if self.state == 'up': self.form_error = error_message
# --- End of Counter Classes ---

# --- Form Correction ---
# (Unchanged)
def check_form(exercise, landmarks, rep_counter):
    try:
        if exercise == 'squat' and rep_counter:
            shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
            torso_angle = calculate_angle(shoulder, hip, knee)
            if rep_counter.state == 'down' and torso_angle < 60: return "Back not straight"
        elif exercise == 'plank':
            shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].z]
            hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].z]
            ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].z]
            body_angle = calculate_angle(shoulder, hip, ankle)
            if body_angle < 140: return "Hips sagging"
            if body_angle > 175: return "Hips too high"
    except Exception as e: pass
    return None

# --- Score Saving ---
# (Unchanged)
SCORE_FILE = "scores.csv"
def save_score(name, final_score):
    headers = ['Name', 'Score', 'Timestamp']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name_standardized = name.strip().lower()
    try:
        if os.path.isfile(SCORE_FILE):
            try:
                df = pd.read_csv(SCORE_FILE)
                if df.empty and not list(df.columns) == headers:
                    df = pd.DataFrame(columns=headers)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=headers)
        else:
            df = pd.DataFrame(columns=headers)
        
        df['Name_std'] = df['Name'].astype(str).str.strip().str.lower()
        if name_standardized in df['Name_std'].values:
            player_index = df[df['Name_std'] == name_standardized].index[0]
            current_score = pd.to_numeric(df.loc[player_index, 'Score'], errors='coerce').fillna(0)
            new_score = current_score + final_score
            df.loc[player_index, 'Score'] = new_score
            df.loc[player_index, 'Timestamp'] = timestamp
            print(f"Updated score for {df.loc[player_index, 'Name']}. New total: {new_score}")
        else:
            new_row = pd.DataFrame([{'Name': name, 'Score': final_score, 'Timestamp': timestamp}])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Added new player {name} with score: {final_score}")
        if 'Name_std' in df.columns:
            df = df.drop(columns=['Name_std'])
        df.to_csv(SCORE_FILE, index=False)
    except Exception as e:
        print(f"Error saving or updating score: {e}")
# --- End of save_score function ---


# --- Main Game Engine Class ---
class GameEngine:
    def __init__(self, player_name, weight=0):
        print("Initializing GameEngine...")
        self.player_name = player_name
        self.weight = weight
        self.model = load_model('exercise_classifier_conv1d_lstm.h5')
        with open('label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)
        self.mp_solutions = mp.solutions
        self.mp_pose = self.mp_solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = self.mp_solutions.drawing_utils
        self.purple_connection_spec = self.mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2)
        self.purple_landmark_spec = self.mp_drawing.DrawingSpec(color=(128, 0, 128), circle_radius=2, thickness=2)
        self.SEQ_LENGTH = 60
        self.NUM_BODY_LANDMARKS = 22
        self.NUM_FEATURES = self.NUM_BODY_LANDMARKS * 3
        self.PREDICTION_THRESHOLD = 0.6
        self.PREDICTION_INTERVAL = 0.2
        self.VISIBILITY_THRESHOLD = 0.5
        self.HISTORY_LEN = 15
        self.DEFAULT_CONSENSUS_COUNT = 7
        self.FAST_CHANGE_COUNT = 4
        self.SLOW_TO_IDLE_COUNT = 13
        self.LATERAL_TO_IDLE_COUNT = 14
        self.EXERCISE_GRACE_PERIOD = 1.5
        self.PLANK_TO_IDLE_COUNT = 7
        self.stable_prediction = "idle"
        self.landmark_sequence = deque(maxlen=self.SEQ_LENGTH)
        self.prediction_history = deque(maxlen=self.HISTORY_LEN)
        self.last_prediction_time = time.time()
        self.score = 0
        self.form_feedback = ""
        self.feedback_display_time = 0
        self.rep_counter_config = {
            'squat': {'class': RepCounter, 'params': {'down_threshold': 90, 'up_threshold': 160, 'exercise_name': 'squat'}, 'is_weighted': True},
            'push-up': {'class': RepCounter, 'params': {'down_threshold': 110, 'up_threshold': 160, 'exercise_name': 'push-up'}, 'is_weighted': False},
            'barbell biceps curl': {'class': RepCounter, 'params': {'down_threshold': 60, 'up_threshold': 150, 'exercise_name': 'bicep curl'}, 'is_weighted': True},
            'hammer curl': {'class': RepCounter, 'params': {'down_threshold': 100, 'up_threshold': 140, 'exercise_name': 'hammer curl'}, 'is_weighted': True},
            'lateral raise': {'class': RepCounterInverted, 'params': {'down_threshold': 30, 'up_threshold': 80, 'exercise_name': 'lateral raise'}, 'is_weighted': True},
        }
        self.current_rep_counter = None
        self.plank_timer_start = None
        self.plank_grace_period_start = None
        self.last_plank_point_time = None
        self.grace_period_active_for = None
        self.grace_period_end_time = 0
        
        self.plank_90s_bonus_awarded = False # <<< --- CHANGED ---
        
        self.pause_start_time = None # For pause logic
        
        print("GameEngine Initialized.")

    # --- Pause/Resume Methods ---
    def pause(self):
        if self.pause_start_time is None: # Only pause if not already paused
            self.pause_start_time = time.time()
            print("GameEngine Paused")

    def unpause(self):
        if self.pause_start_time is not None: # Only unpause if paused
            pause_duration = time.time() - self.pause_start_time
            print(f"GameEngine Resumed after {pause_duration:.2f}s")
            
            # Add pause duration to all time-sensitive variables
            if self.last_prediction_time: self.last_prediction_time += pause_duration
            if self.plank_timer_start: self.plank_timer_start += pause_duration
            if self.plank_grace_period_start: self.plank_grace_period_start += pause_duration
            if self.last_plank_point_time: self.last_plank_point_time += pause_duration
            if self.grace_period_end_time > 0: self.grace_period_end_time += pause_duration
            if self.feedback_display_time > 0: self.feedback_display_time += pause_duration
            
            self.pause_start_time = None # Reset pause time
    # --- End Pause/Resume ---

    def process_frame(self, frame):
        # (Grace period check)
        if self.grace_period_active_for is not None and time.time() > self.grace_period_end_time:
            self.current_rep_counter = None
            self.grace_period_active_for = None
        
        # (Pose estimation & normalization)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        current_landmarks_flat = np.zeros(self.NUM_FEATURES)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.purple_landmark_spec, self.purple_connection_spec)
            landmarks_list = results.pose_landmarks.landmark; frame_landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list]); lm_left_hip = landmarks_list[self.mp_pose.PoseLandmark.LEFT_HIP.value]; lm_right_hip = landmarks_list[self.mp_pose.PoseLandmark.RIGHT_HIP.value]; lm_left_shoulder = landmarks_list[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]; lm_right_shoulder = landmarks_list[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]; hip_visibility_ok = (lm_left_hip.visibility > self.VISIBILITY_THRESHOLD and lm_right_hip.visibility > self.VISIBILITY_THRESHOLD); shoulder_visibility_ok = (lm_left_shoulder.visibility > self.VISIBILITY_THRESHOLD and lm_right_shoulder.visibility > self.VISIBILITY_THRESHOLD)
            if hip_visibility_ok: center_point = (frame_landmarks_np[self.mp_pose.PoseLandmark.LEFT_HIP.value] + frame_landmarks_np[self.mp_pose.PoseLandmark.RIGHT_HIP.value]) / 2.0; normalized_landmarks = frame_landmarks_np - center_point; current_landmarks_flat = normalized_landmarks[11:].flatten()
            elif shoulder_visibility_ok: center_point = (frame_landmarks_np[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value] + frame_landmarks_np[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]) / 2.0; normalized_landmarks = frame_landmarks_np - center_point; current_landmarks_flat = normalized_landmarks[11:].flatten()
        self.landmark_sequence.append(current_landmarks_flat)
        
        # (Classification & Stability Logic)
        if (time.time() - self.last_prediction_time > self.PREDICTION_INTERVAL) and len(self.landmark_sequence) == self.SEQ_LENGTH:
            self.last_prediction_time = time.time()
            landmark_list = list(self.landmark_sequence); padded_sequence = pad_sequences([landmark_list], maxlen=self.SEQ_LENGTH, dtype='float32', padding='pre')
            padded_sequence = padded_sequence.reshape(1, self.SEQ_LENGTH, self.NUM_FEATURES); prediction = self.model.predict(padded_sequence, verbose=0)[0]; predicted_class_index = np.argmax(prediction); prediction_probability = prediction[predicted_class_index]; current_prediction = "idle"
            if prediction_probability > self.PREDICTION_THRESHOLD: current_prediction = self.le.classes_[predicted_class_index]
            self.prediction_history.append(current_prediction)
            if len(self.prediction_history) == self.HISTORY_LEN:
                try:
                    majority_prediction = max(set(self.prediction_history), key=list(self.prediction_history).count)
                    consensus_count = list(self.prediction_history).count(majority_prediction)
                    if majority_prediction != self.stable_prediction:
                        required_count = self.DEFAULT_CONSENSUS_COUNT
                        if self.stable_prediction != 'idle' and majority_prediction != 'idle': required_count = self.FAST_CHANGE_COUNT
                        elif self.stable_prediction != 'idle' and majority_prediction == 'idle':
                            if self.stable_prediction == 'lateral raise': required_count = self.LATERAL_TO_IDLE_COUNT
                            elif self.stable_prediction == 'plank': required_count = self.PLANK_TO_IDLE_COUNT
                            else: required_count = self.SLOW_TO_IDLE_COUNT
                        if consensus_count >= required_count:
                            previous_stable_prediction = self.stable_prediction; self.stable_prediction = majority_prediction; self.form_feedback = ""
                            if self.stable_prediction in self.rep_counter_config: self.grace_period_active_for = None
                            elif previous_stable_prediction in self.rep_counter_config and self.stable_prediction == 'idle':
                                self.grace_period_active_for = previous_stable_prediction; self.grace_period_end_time = time.time() + self.EXERCISE_GRACE_PERIOD
                            if self.stable_prediction != 'plank': self.plank_timer_start = None; self.plank_grace_period_start = None
                            elif self.stable_prediction == 'plank' and previous_stable_prediction != 'plank': self.plank_grace_period_start = time.time(); self.plank_timer_start = None; self.last_plank_point_time = None
                            if self.grace_period_active_for is None:
                                if self.stable_prediction in self.rep_counter_config:
                                    config = self.rep_counter_config[self.stable_prediction]
                                    current_weight = self.weight if config.get('is_weighted', False) else 0
                                    params = config['params'].copy()
                                    params['weight'] = current_weight
                                    self.current_rep_counter = config['class'](**params)
                                else: self.current_rep_counter = None
                except ValueError: pass
        
        # --- Exercise-Specific Logic ---
        display_text = f"{self.stable_prediction}"
        exercise_to_process = self.grace_period_active_for if self.grace_period_active_for is not None else self.stable_prediction
        reps_text = "N/A"; current_feedback = ""
        
        if exercise_to_process == 'plank':
            plank_form_ok = True
            if results.pose_landmarks:
                feedback = check_form('plank', results.pose_landmarks.landmark, None)
                if feedback: current_feedback = feedback; plank_form_ok = False
            if self.plank_grace_period_start is not None:
                if self.plank_timer_start is None:
                    grace_time_elapsed = time.time() - self.plank_grace_period_start
                    if grace_time_elapsed >= 5.0:
                        self.plank_timer_start = time.time(); self.last_plank_point_time = time.time(); plank_time_elapsed = 0.0
                        self.plank_90s_bonus_awarded = False # <<< --- CHANGED: Reset bonus flag ---
                        reps_text = f"{plank_time_elapsed:.1f}s"
                    else:
                        countdown = 5.0 - grace_time_elapsed
                        reps_text = f"Start in: {countdown:.0f}s"
                else:
                    current_time = time.time()
                    plank_time_elapsed = current_time - self.plank_timer_start
                    reps_text = f"{plank_time_elapsed:.1f}s"
                    
                    # <<< --- CHANGED: Add 90s bonus logic ---
                    if plank_time_elapsed >= 90.0 and not self.plank_90s_bonus_awarded:
                        self.score += 30
                        self.plank_90s_bonus_awarded = True
                        current_feedback = "+30 Bonus!" 
                        print("Awarded 90-second plank bonus!")
                    # <<< --- END CHANGE ---
                    
                    if current_time - (self.last_plank_point_time or self.plank_timer_start) >= 1.0:
                        self.score += 2 if plank_form_ok else 1
                        self.last_plank_point_time = current_time 

        elif exercise_to_process in self.rep_counter_config and self.current_rep_counter:
            exercise_name_for_logic = exercise_to_process
            reps_text = str(self.current_rep_counter.count)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark; angle = 0; points_this_update = 0
                try:
                    if exercise_name_for_logic == 'squat':
                        shoulder, hip, knee = (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE); left_vis = lm[hip.value].visibility; right_vis = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility; chosen_hip = hip
                        if right_vis > left_vis and right_vis > self.VISIBILITY_THRESHOLD: shoulder, hip, knee = (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE); chosen_hip = hip
                        if lm[chosen_hip.value].visibility > self.VISIBILITY_THRESHOLD: angle = calculate_angle([lm[shoulder.value].x, lm[shoulder.value].y, lm[shoulder.value].z],[lm[hip.value].x, lm[hip.value].y, lm[hip.value].z],[lm[knee.value].x, lm[knee.value].y, lm[knee.value].z])
                    elif exercise_name_for_logic in ['barbell biceps curl', 'hammer curl', 'push-up']:
                        shoulder, elbow, wrist = (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST); left_vis = lm[elbow.value].visibility; right_vis = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility; chosen_elbow = elbow
                        if right_vis > left_vis and right_vis > self.VISIBILITY_THRESHOLD: shoulder, elbow, wrist = (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST); chosen_elbow = elbow
                        elbow_visibility = lm[chosen_elbow.value].visibility
                        if elbow_visibility > self.VISIBILITY_THRESHOLD: angle = calculate_angle([lm[shoulder.value].x, lm[shoulder.value].y, lm[shoulder.value].z],[lm[elbow.value].x, lm[elbow.value].y, lm[elbow.value].z],[lm[wrist.value].x, lm[wrist.value].y, lm[wrist.value].z])
                    elif exercise_name_for_logic == 'lateral raise':
                        hip, shoulder, elbow = (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmask.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW); left_vis = lm[shoulder.value].visibility; right_vis = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility; chosen_shoulder = shoulder
                        if right_vis > left_vis and right_vis > self.VISIBILITY_THRESHOLD: hip, shoulder, elbow = (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW); chosen_shoulder = shoulder
                        if lm[chosen_shoulder.value].visibility > self.VISIBILITY_THRESHOLD: angle = calculate_angle([lm[hip.value].x, lm[hip.value].y, lm[hip.value].z],[lm[shoulder.value].x, lm[shoulder.value].y, lm[shoulder.value].z],[lm[elbow.value].x, lm[elbow.value].y, lm[elbow.value].z])
                    
                    feedback = check_form(exercise_name_for_logic, lm, self.current_rep_counter)
                    if feedback: self.current_rep_counter.log_form_error(feedback)
                    if angle > 0:
                        rep_status, points_this_update = self.current_rep_counter.update(angle); self.score += points_this_update
                        if rep_status == "rep_counted": print(f"{exercise_name_for_logic.replace('_', ' ').title()} Rep Counted! Total: {self.current_rep_counter.count}")
                        elif rep_status == "partial_rep":
                            feedback_text = "Go deeper" if exercise_name_for_logic != 'lateral raise' else "Raise higher"; current_feedback = feedback_text
                        elif rep_status and "form_error:" in rep_status:
                            current_feedback = rep_status.split(":", 1)[1]
                except Exception as e: pass

        # --- Draw on frame ---
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {self.score}", (frame.shape[1] - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        if self.current_rep_counter: reps_text = str(self.current_rep_counter.count)
        cv2.putText(frame, f"Reps: {reps_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if current_feedback: self.form_feedback = current_feedback; self.feedback_display_time = time.time()
        if self.form_feedback and (time.time() - self.feedback_display_time < 2.0):
            cv2.putText(frame, self.form_feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else: self.form_feedback = ""
        
        return frame, self.stable_prediction, reps_text, self.score, self.form_feedback
    
    def save_final_score(self):
        print(f"Saving final score for {self.player_name}: {self.score}")
        save_score(self.player_name, self.score) # Uses the new pandas function

    def close(self):
        self.pose.close()
        print("GameEngine closed.")