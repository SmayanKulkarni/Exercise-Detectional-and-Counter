import streamlit as st
import os
import pandas as pd

# --- Page Config (MUST be the first Streamlit command) ---
st.set_page_config(page_title="XpBoost", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Button Styling */
    div[data-testid="stButton"] > button { width: 100%; }
    div[data-testid="stButton"] > button:contains("Start Game") {
        background-color: #4CAF50; color: white; border-color: #4CAF50;
    }
    div[data-testid="stButton"] > button:contains("Stop Game / Cancel") {
        background-color: #FF4B4B; color: white; border-color: #FF4B4B;
    }
    div[data-testid="stButton"] > button:contains("Confirm and Save Score") {
        background-color: #4CAF50; color: white; border-color: #4CAF50;
    }

    /* --- GIF Gallery Styling --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        padding: 1em;
        height: 100%; /* Make cards in a row the same height */
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        text-align: center;
    }
    /* This block ensures animated GIFs play and are sized correctly */
    div[data-testid="stImage"] {
        height: 250px; /* Fixed height for all image containers */
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stImage"] img {
        object-fit: cover; /* "Zoom to fill" the container */
        height: 100%;
        width: 100%;
    }
    /* Style for the caption text (larger) */
    div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
        min-height: 60px;
        font-style: italic;
        font-size: 1.05em; /* Make it slightly larger */
        font-weight: 500; /* Make it a bit bolder */
    }
</style>
""", unsafe_allow_html=True)
# --- End Custom CSS ---

# --- Home Page Content ---
st.title("🏋️ Posture Perfect 🏆")
st.info("Check out the available exercises below. When you're ready, click **'Start Game'** in the sidebar on the left!")

# --- Show Available Exercises ---
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

# --- Point system descriptions (to match your game_engine.py) ---
point_descriptions = {
    "Squat": "+5 points (+ weight bonus) per deep rep. Shallow reps or bad form: -3 points.",
    "Push-up": "+5 points per deep rep (no weight bonus). Shallow reps: -3 points.",
    "Hammer Curl": "+5 points (+ weight bonus) per correct rep. Partial reps: -3 points.",
    "Lateral Raise": "+5 points (+ weight bonus) per high rep. Partial reps: -3 points.",
    "Plank": "+2 points/sec for good form. +1 point/sec if hips sag or pike."
}

col1, col2, col3, col4, col5 = st.columns(5)

# Helper function to display an exercise in a "card"
def show_exercise(column, name, path):
    with column:
        with st.container(border=True):
            st.subheader(name)
            if os.path.isfile(path):
                # Using use_container_width=True with the CSS above
                # is the correct way to make animated GIFs play and size properly
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Image not found: {path}")
            
            description = point_descriptions.get(name, "Earn points for correct reps.")
            st.caption(description)

# Populate the grid
show_exercise(col1, "Squat", exercise_images["Squat"])
show_exercise(col2, "Push-up", exercise_images["Push-up"])
show_exercise(col3, "Hammer Curl", exercise_images["Hammer Curl"])
show_exercise(col4, "Lateral Raise", exercise_images["Lateral Raise"])
show_exercise(col5, "Plank", exercise_images["Plank"])

# --- Display Leaderboard ---
SCORE_FILE = "scores.csv"
if os.path.isfile(SCORE_FILE):
    st.divider()
    st.header("Leaderboard")
    try:
        # Add error handling for an empty CSV
        df = pd.read_csv(SCORE_FILE)
        if not df.empty:
            df_sorted = df.sort_values(by="Score", ascending=False).head(10)
            st.dataframe(df_sorted, hide_index=True, use_container_width=True)
        else:
            st.info("Leaderboard is empty! Be the first to play.")
    except pd.errors.EmptyDataError:
        st.info("Leaderboard is empty! Be the first to play.")
    except Exception as e:
        st.error(f"Could not load leaderboard: {e}")