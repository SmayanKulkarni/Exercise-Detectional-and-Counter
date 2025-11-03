import streamlit as st
import cv2
import os
import pandas as pd
from game_engine import GameEngine, save_score # Import save_score directly

# --- Page Config ---
st.set_page_config(page_title="Fitness Challenge", layout="wide")

# --- NEW: Custom CSS Styling ---
st.markdown("""
<style>
    /* Make buttons fill their columns */
    div[data-testid="stButton"] > button {
        width: 100%;
    }

    /* Make "Start Game" button green */
    div[data-testid="stButton"] > button:contains("Start Game") {
        background-color: #4CAF50; /* Green */
        color: white;
        border-color: #4CAF50;
    }

    /* Make "Stop Game / Cancel" button red */
    div[data-testid="stButton"] > button:contains("Stop Game / Cancel") {
        background-color: #FF4B4B; /* Red */
        color: white;
        border-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)
# --- End Custom CSS ---


st.title("🏋️ Fitness Challenge Event 🏆")

# --- Session State Initialization ---
# (Unchanged)
if 'run_game' not in st.session_state:
    st.session_state.run_game = False
if 'player_name' not in st.session_state:
    st.session_state.player_name = ""
if 'game_engine' not in st.session_state:
    st.session_state.game_engine = None
if 'confirm_score' not in st.session_state:
    st.session_state.confirm_score = False
# ---

# --- 1. Player Name and Controls ---
st.header("Player Setup")

is_disabled = st.session_state.run_game or st.session_state.confirm_score

col1_setup, col2_setup = st.columns(2)
with col1_setup:
    player_name_input = st.text_input(
        "Enter your name:", 
        st.session_state.player_name, 
        disabled=is_disabled
    )
with col2_setup:
    weight_input = st.number_input(
        "Dumbbell Weight (per hand, in kg)", 
        min_value=0, max_value=50, value=0, step=1, 
        disabled=is_disabled
    )

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Game", type="primary", disabled=is_disabled):
        if player_name_input:
            st.session_state.player_name = player_name_input
            st.session_state.game_engine = GameEngine(player_name_input, weight_input) 
            st.session_state.run_game = True
            st.session_state.confirm_score = False
            st.rerun()
        else:
            st.warning("Please enter a name to start.")

with col2:
    if st.button("Stop Game / Cancel", disabled=not (st.session_state.run_game or st.session_state.confirm_score)):
        if st.session_state.run_game and st.session_state.game_engine is not None:
            st.session_state.run_game = False
            st.session_state.confirm_score = True
        else:
            st.session_state.run_game = False
            st.session_state.confirm_score = False
            st.session_state.game_engine = None
            st.session_state.player_name = ""
        st.rerun()

st.divider()

# --- 2. Display Area ---
st.sidebar.header("🏆 Live Stats 🏆")
# --- UPDATED: Added emojis to metrics ---
score_placeholder = st.sidebar.metric("🏆 Score", 0)
reps_placeholder = st.sidebar.metric("💪 Reps/Time", "N/A")
feedback_placeholder = st.sidebar.empty()
# ---

st_frame = st.empty()

if st.session_state.run_game and st.session_state.game_engine is not None:
    # --- Main Game Loop ---
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check permissions.")
    else:
        while st.session_state.run_game:
            ret, frame = cap.read()
            if not ret:
                st.write("Webcam feed ended.")
                st.session_state.run_game = False
                break
            
            frame = cv2.flip(frame, 1)
            
            processed_frame, pred, reps, score, feedback = st.session_state.game_engine.process_frame(frame)
            
            score_placeholder.metric("🏆 Score", score)
            reps_placeholder.metric(f"💪 {pred.replace('_', ' ').title()}", reps)
            
            # --- UPDATED: Bolder feedback ---
            if feedback:
                feedback_placeholder.warning(f"**{feedback}!**")
            else:
                feedback_placeholder.empty()

            # --- UPDATED: use_container_width ---
            st_frame.image(processed_frame, channels="BGR", use_container_width=True)

        cap.release()
        print("Webcam released.")
        st.rerun()

# --- Confirmation Screen ---
elif st.session_state.confirm_score and st.session_state.game_engine is not None:
    st.header("Confirm Final Score")
    st.subheader(f"Player: {st.session_state.player_name}")

    calc_score = st.session_state.game_engine.score
    st.metric("Calculated Score", calc_score)

    modification_points = st.number_input("Score Modification (e.g., -20 or 50)", value=0, step=5)
    
    final_score = calc_score + modification_points
    st.metric("Final Score to be Saved", final_score, delta=f"{modification_points:+} points")

    if st.button("Confirm and Save Score", type="primary"):
        save_score(st.session_state.player_name, final_score)
        st.success(f"Final score for {st.session_state.player_name} saved: {final_score}")
        
        st.session_state.run_game = False
        st.session_state.confirm_score = False
        st.session_state.game_engine = None
        st.session_state.player_name = ""
        st.rerun()

# --- Home Screen ---
else:
    st.info("Enter your name, set your weight, and click 'Start Game' to begin!")

    # --- UPDATED: Show Available Exercises using Containers ---
    st.divider()
    st.header("Available Exercises")
    
    image_folder = "images" 
    exercise_images = {
        "Squat": f"{image_folder}/squat.gif",
        "Push-up": f"{image_folder}/pushup.gif",
        "Hammer Curl": f"{image_folder}/hammer_curl.gif",
        "Lateral Raise": f"{image_folder}/lateral_raise.gif",
        "Plank": f"{image_folder}/plank.gif",
    }
    
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    # Helper function to display an exercise in a "card"
    def show_exercise(column, name, path):
        with column:
            # Use st.container(border=True) to create a styled card
            with st.container(border=True):
                st.subheader(name)
                if os.path.isfile(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning(f"Image not found: {path}")

    # Populate the grid
    show_exercise(col1, "Squat", exercise_images["Squat"])
    show_exercise(col2, "Push-up", exercise_images["Push-up"])
    show_exercise(col4, "Hammer Curl", exercise_images["Hammer Curl"])
    show_exercise(col5, "Lateral Raise", exercise_images["Lateral Raise"])
    show_exercise(col6, "Plank", exercise_images["Plank"])
    # --- End of New Section ---

    # Display Leaderboard
    SCORE_FILE = "scores.csv"
    if os.path.isfile(SCORE_FILE):
        st.divider()
        st.header("Leaderboard")
        try:
            df = pd.read_csv(SCORE_FILE)
            df_sorted = df.sort_values(by="Score", ascending=False).head(10)
            # --- UPDATED: use_container_width ---
            st.dataframe(df_sorted, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load leaderboard: {e}")