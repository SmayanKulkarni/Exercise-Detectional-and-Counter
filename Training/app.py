import streamlit as st
import cv2
import os
import pandas as pd
from game_engine import GameEngine # Import our class

# --- Page Config ---
st.set_page_config(page_title="Fitness Challenge", layout="wide")
st.title("🏋️ Fitness Challenge Event 🏆")

# --- Session State Initialization ---
if 'run_game' not in st.session_state:
    st.session_state.run_game = False
if 'player_name' not in st.session_state:
    st.session_state.player_name = ""
if 'game_engine' not in st.session_state:
    st.session_state.game_engine = None
# ---

# --- 1. Player Name and Controls ---
st.header("Player Setup")

# --- UPDATED: Added weight input ---
col1_setup, col2_setup = st.columns(2)
with col1_setup:
    player_name_input = st.text_input("Enter your name:", st.session_state.player_name)
with col2_setup:
    weight_input = st.number_input("Dumbbell Weight (per hand, in kg)", min_value=0, max_value=50, value=0, step=1)
# ---

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Game", type="primary"):
        if player_name_input:
            st.session_state.player_name = player_name_input
            # --- UPDATED: Pass the weight to the engine ---
            st.session_state.game_engine = GameEngine(player_name_input, weight_input) 
            st.session_state.run_game = True
            st.rerun() # Rerun to start the game loop
        else:
            st.warning("Please enter a name to start.")

with col2:
    if st.button("Stop Game"):
        if st.session_state.run_game and st.session_state.game_engine is not None:
            # Save the score using the engine's method
            st.session_state.game_engine.save_final_score()
            st.success(f"Score for {st.session_state.player_name} saved: {st.session_state.game_engine.score}")
        
        # (Reset logic - unchanged)
        st.session_state.run_game = False
        st.session_state.player_name = ""
        st.session_state.game_engine = None
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
    cap = cv2.VideoCapture("/home/smayan/Downloads/Exercises/Testing Videos/plank.mp4")
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check permissions.")
    else:
        while st.session_state.run_game: # Loop while 'run_game' is True
            ret, frame = cap.read()
            if not ret:
                st.write("Webcam feed ended.")
                st.session_state.run_game = False
                break
            
            frame = cv2.flip(frame, 1) # Flip for mirror view
            
            # Process the frame using the game engine
            processed_frame, pred, reps, score, feedback = st.session_state.game_engine.process_frame(frame)
            
            # --- Display Information ---
            score_placeholder.metric("Score", score)
            reps_placeholder.metric(f"{pred.replace('_', ' ').title()} Reps/Time", reps)
            if feedback:
                feedback_placeholder.warning(feedback)
            else:
                feedback_placeholder.empty()

            st_frame.image(processed_frame, channels="BGR", use_container_width=True) # Updated param

        cap.release()
        print("Webcam released.")
        
        # (Cleanup logic - unchanged)
        if st.session_state.game_engine is not None:
            st.session_state.game_engine.save_final_score()
            st.success(f"Session stopped. Final score for {st.session_state.player_name}: {st.session_state.game_engine.score}")
            st.session_state.game_engine = None
            st.session_state.player_name = ""
        
        st.session_state.run_game = False
        st.rerun()

else:
    st.info("Enter your name, set your weight, and click 'Start Game' to begin!")

    # (Leaderboard logic - unchanged)
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