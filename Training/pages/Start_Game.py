import streamlit as st
import cv2
import os
import pandas as pd
import time
from game_engine import GameEngine, save_score # Import our game logic

# --- Page Config ---
st.set_page_config(page_title="Start Game", layout="wide")

# --- Inject CSS (so buttons are styled on this page too) ---
st.markdown("""
<style>
    div[data-testid="stButton"] > button { width: 100%; }
    div[data-testid="stButton"] > button:contains("Start Game") {
        background-color: #4CAF50; color: white; border-color: #4CAF50;
    }
    div[data-testid="stButton"] > button:contains("Pause") {
        background-color: #FFA500; color: white; border-color: #FFA500;
    }
    div[data-testid="stButton"] > button:contains("Resume") {
        background-color: #4CAF50; color: white; border-color: #4CAF50;
    }
    div[data-testid="stButton"] > button:contains("Stop Game / Cancel") {
        background-color: #FF4B4B; color: white; border-color: #FF4B4B;
    }
    div[data-testid="stButton"] > button:contains("Confirm and Save Score") {
        background-color: #4CAF50; color: white; border-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


st.title("🎮 Game On!")

# --- Session State Initialization ---
if 'run_game' not in st.session_state:
    st.session_state.run_game = False
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'player_name' not in st.session_state:
    st.session_state.player_name = ""
if 'game_engine' not in st.session_state:
    st.session_state.game_engine = None
if 'confirm_score' not in st.session_state:
    st.session_state.confirm_score = False
if 'last_frame' not in st.session_state: # To freeze frame on pause
    st.session_state.last_frame = None
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

# --- UPDATED: 3-Button Layout ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start Game", type="primary", disabled=is_disabled):
        if player_name_input:
            st.session_state.player_name = player_name_input.strip() # Store the original name
            # Pass the name to the engine
            st.session_state.game_engine = GameEngine(st.session_state.player_name, weight_input) 
            st.session_state.run_game = True
            st.session_state.is_paused = False
            st.session_state.confirm_score = False
            st.rerun()
        else:
            st.warning("Please enter a name to start.")

with col2:
    # Pause/Resume Button
    pause_text = "Resume" if st.session_state.is_paused else "Pause"
    if st.button(pause_text, disabled=not st.session_state.run_game or st.session_state.confirm_score):
        st.session_state.is_paused = not st.session_state.is_paused # Toggle state
        if st.session_state.is_paused:
            st.session_state.game_engine.pause()
        else:
            st.session_state.game_engine.unpause()
        st.rerun()

with col3:
    # Stop/Cancel Button
    stop_text = "Stop Game" if st.session_state.run_game else "Cancel"
    if st.button(stop_text, disabled=not (st.session_state.run_game or st.session_state.confirm_score)):
        if st.session_state.run_game and st.session_state.game_engine is not None:
            st.session_state.run_game = False
            st.session_state.is_paused = False
            st.session_state.confirm_score = True # Go to confirm screen
        else: # Cancel from setup or confirm screen
            st.session_state.run_game = False
            st.session_state.confirm_score = False
            if st.session_state.game_engine:
                st.session_state.game_engine.close()
            st.session_state.game_engine = None
            st.session_state.player_name = ""
        st.rerun()

st.divider()

# --- 2. Display Area ---
st.sidebar.header("🏆 Live Stats 🏆")
# Set default values for when game_engine is None
score_val = st.session_state.game_engine.score if st.session_state.game_engine else 0
pred_val = st.session_state.game_engine.stable_prediction if st.session_state.game_engine else "N/A"
reps_val = "N/A"
if st.session_state.game_engine and st.session_state.game_engine.current_rep_counter:
    reps_val = st.session_state.game_engine.current_rep_counter.count

score_placeholder = st.sidebar.metric("🏆 Score", score_val)
reps_placeholder = st.sidebar.metric(f"💪 {pred_val.replace('_', ' ').title()}", reps_val)
feedback_placeholder = st.sidebar.empty()

st_frame = st.empty()

if st.session_state.run_game and st.session_state.game_engine is not None:
    # --- Main Game Loop ---
    cap = cv2.VideoCapture("/home/smayan/Downloads/Exercises/Testing Videos/pushup.webm")
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check permissions.")
    else:
        while st.session_state.run_game:
            # --- UPDATED: Pause logic ---
            if not st.session_state.is_paused:
                ret, frame = cap.read()
                if not ret:
                    st.write("Webcam feed ended.")
                    st.session_state.run_game = False
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process the frame
                processed_frame, pred, reps, score, feedback = st.session_state.game_engine.process_frame(frame)
                
                # Store the latest processed frame
                st.session_state.last_frame = processed_frame
                
                # Update sidebar
                score_placeholder.metric("🏆 Score", score)
                reps_placeholder.metric(f"💪 {pred.replace('_', ' ').title()}", reps)
                if feedback:
                    feedback_placeholder.warning(f"**{feedback}!**")
                else:
                    feedback_placeholder.empty()
            
            # --- Always display a frame ---
            if st.session_state.last_frame is not None:
                st_frame.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
            else:
                st_frame.info("Waiting for webcam...")
            
            # If paused, show a message
            if st.session_state.is_paused:
                feedback_placeholder.info("Game Paused ⏸️")
            
            # This is a small sleep to prevent the loop from hogging the CPU when paused
            time.sleep(0.01) 

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
        # Save the score using the original name
        save_score(st.session_state.player_name, final_score)
        st.success(f"Final score for {st.session_state.player_name} saved: {final_score}")
        
        st.session_state.game_engine.close() # Clean up engine
        st.session_state.run_game = False
        st.session_state.confirm_score = False
        st.session_state.game_engine = None
        st.session_state.player_name = ""
        
        st.switch_page("app.py") # Go back to home page

# --- Home Screen ---
else:
    st.info("Enter your name and click 'Start Game' to begin!")