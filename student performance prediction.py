import streamlit as st
import numpy as np
import pickle

# Load trained model
# NOTE: The model was trained on 5 raw features directly (without polynomial transformation)
try:
    model = pickle.load(open("best_model.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 1.1rem;
            padding: 0.6rem 2rem;
            border-radius: 8px;
            width: 100%;
        }
        .stButton>button:hover { background-color: #45a049; }
        .grade-box {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            margin-top: 1rem;
        }
        .grade-letter { font-size: 4rem; font-weight: 900; }
        .grade-score  { font-size: 1.3rem; opacity: 0.85; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Student Grade Prediction System")
st.write("Predict a student's **final score** and **grade** using a Machine Learning model.")

st.divider()

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info(
"""
This ML model predicts a student's **final score & grade** based on:

• 📚 Weekly Self-Study Hours  
• 🏫 Attendance Percentage  
• 🙋 Class Participation  

_Higher study hours = higher predicted score!_
"""
)
st.sidebar.markdown("---")
st.sidebar.subheader("Grade Scale")
st.sidebar.markdown(
    "| Grade | Score |\n|-------|-------|\n| A | ≥ 90 |\n| B | ≥ 75 |\n| C | ≥ 60 |\n| D | ≥ 50 |\n| F | < 50 |"
)

# Input Section
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Weekly Self Study Hours", 0, 40, 10,
                            help="How many hours per week the student studies on their own. More hours = better score!")
    attendance = st.slider("🏫 Attendance Percentage (%)", 0, 100, 75,
                           help="Percentage of classes attended.")

with col2:
    participation = st.slider("🙋 Class Participation (0–10)", 0, 10, 5,
                              help="Level of active participation in class (0 = none, 10 = very active).")

st.divider()


def get_grade(score: float) -> tuple[str, str]:
    """Return (letter_grade, color) based on score."""
    if score >= 90:
        return "A", "#00c853"
    elif score >= 75:
        return "B", "#64dd17"
    elif score >= 60:
        return "C", "#ffd600"
    elif score >= 50:
        return "D", "#ff6d00"
    else:
        return "F", "#d50000"


# Prediction Button
if st.button("🚀 Predict Grade"):

    # Build feature array — 3 features: study_hours, attendance, participation
    features = np.array([[study_hours, attendance, participation]])

    try:
        predicted_score = model.predict(features)[0]
        predicted_score = float(np.clip(predicted_score, 0, 100))  # clamp to [0, 100]
        grade, color = get_grade(predicted_score)

        # Display results
        st.success("✅ Prediction Complete!")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("📊 Predicted Score", f"{predicted_score:.1f} / 100")
        col_b.metric("🏅 Grade", grade)

        # Determine performance label
        if grade == "A":
            perf_label = "🌟 Outstanding"
        elif grade == "B":
            perf_label = "😊 Good"
        elif grade == "C":
            perf_label = "😐 Average"
        elif grade == "D":
            perf_label = "⚠️ Below Average"
        else:
            perf_label = "❌ Failing"

        col_c.metric("📈 Performance", perf_label)

        # Visual grade card
        st.markdown(f"""
            <div class="grade-box" style="border: 3px solid {color};">
                <div class="grade-letter" style="color:{color};">{grade}</div>
                <div class="grade-score">Predicted Score: <strong>{predicted_score:.1f}</strong> / 100</div>
                <div style="margin-top:0.5rem; font-size:1rem; opacity:0.75;">{perf_label}</div>
            </div>
        """, unsafe_allow_html=True)

        st.balloons()

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Make sure `best_model.pkl` is in the same folder as this script.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Scikit-learn Machine Learning")