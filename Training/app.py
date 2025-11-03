import streamlit as st
import cv2
import os
import pandas as pd
from game_engine import GameEngine, save_score # Import save_score directly

# --- Page Config ---
st.set_page_config(page_title="Fitness Challenge", layout="wide")
st.title("🏋️ Posture Perfect 🏆")

# --- Session State Initialization ---
if 'run_game' not in st.session_state:
    st.session_state.run_game = False
if 'player_name' not in st.session_state:
    st.session_state.player_name = ""
if 'game_engine' not in st.session_state:
    st.session_state.game_engine = None
if 'confirm_score' not in st.session_state: # NEW: State for confirmation
    st.session_state.confirm_score = False
# ---

# --- 1. Player Name and Controls ---
st.header("Player Setup")

# Disable inputs if game is running or needs confirmation
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
            st.session_state.confirm_score = False # Ensure confirm is off
            st.rerun()
        else:
            st.warning("Please enter a name to start.")

with col2:
    # This button now stops the game OR cancels the confirmation
    if st.button("Stop Game / Cancel"):
        if st.session_state.run_game: # If game is running, stop it and go to confirm
            st.session_state.run_game = False
            st.session_state.confirm_score = True # Go to confirm screen
        else: # If game is not running (on setup or confirm screen), just reset
            st.session_state.run_game = False
            st.session_state.confirm_score = False
            st.session_state.game_engine = None
            st.session_state.player_name = ""
        st.rerun()

st.divider()

# --- 2. Display Area ---
st.sidebar.header("🏆 Live Stats 🏆")
score_placeholder = st.sidebar.metric("Score", 0)
reps_placeholder = st.sidebar.metric("Reps/Time", "N/A")
feedback_placeholder = st.sidebar.empty()

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
            
            frame = cv2.flip(frame, 1) # Flip for mirror view
            
            processed_frame, pred, reps, score, feedback = st.session_state.game_engine.process_frame(frame)
            
            score_placeholder.metric("Score", score)
            reps_placeholder.metric(f"{pred.replace('_', ' ').title()} Reps/Time", reps)
            if feedback:
                feedback_placeholder.warning(feedback)
            else:
                feedback_placeholder.empty()

            st_frame.image(processed_frame, channels="BGR", use_container_width=True)

        cap.release()
        print("Webcam released.")
        # When loop stops, rerun to show the confirmation screen
        st.rerun()

# --- NEW: Confirmation Screen ---
elif st.session_state.confirm_score and st.session_state.game_engine is not None:
    st.header("Confirm Final Score")
    st.subheader(f"Player: {st.session_state.player_name}")

    calc_score = st.session_state.game_engine.score
    st.metric("Calculated Score", calc_score)

    modification_points = st.number_input("Score Modification (e.g., -20 or 50)", value=0, step=5)
    
    final_score = calc_score + modification_points
    st.metric("Final Score to be Saved", final_score)

    if st.button("Confirm and Save Score", type="primary"):
        # Save the *final* score
        save_score(st.session_state.player_name, final_score)
        st.success(f"Final score for {st.session_state.player_name} saved: {final_score}")
        
        # Reset everything for the next player
        st.session_state.run_game = False
        st.session_state.confirm_score = False
        st.session_state.game_engine = None
        st.session_state.player_name = ""
        st.rerun() # Go back to the main setup page

else:
    st.info("Enter your name, set your weight, and click 'Start Game' to begin!")

    # Display Leaderboard on the home screen
    SCORE_FILE = "scores.csv"
    if os.path.isfile(SCORE_FILE):
        st.divider()
        st.header("Leaderboard")
        try:
            df = pd.read_csv(SCORE_FILE)
            df_sorted = df.sort_values(by="Score", ascending=False).head(10)
            st.dataframe(df_sorted, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load leaderboard: {e}")