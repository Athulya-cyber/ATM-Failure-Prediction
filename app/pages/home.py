import streamlit as st
import pandas as pd
import joblib


# -------------------------------------------------
# PAGE TITLE
# -------------------------------------------------

st.title("🏧 ATM Failure Prediction")

st.subheader("Machine Learning Project")

st.write(
    "This application uses a Random Forest Classifier to predict "
    "whether an ATM will fail in the next 48 hours or not."
)

st.markdown("---")


# -------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------

st.subheader("Project Overview")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "Dataset" / "ATM_Maintenance_Cleaned.csv"
FEATURE_PATH = BASE_DIR / "Models" / "feature_cols.joblib"

atm = pd.read_csv("ATM_Maintenance_Cleaned.csv")

feature_cols = joblib.load("feature_cols.joblib")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset Records",
        atm.shape[0]
    )

with col2:
    st.metric(
        "Features Used",
        len(feature_cols)
    )

with col3:
    st.metric(
        "Machine Learning Model",
        "Random Forest"
    )


st.markdown("---")


# -------------------------------------------------
# PROJECT OBJECTIVE
# -------------------------------------------------

st.subheader("🎯 Project Objective")

st.write(
    "The objective of this project is to identify ATMs that are at risk "
    "of failure within the next 48 hours. The prediction can help support "
    "preventive maintenance by identifying potentially high-risk ATMs "
    "before a failure occurs."
)


st.markdown("---")


# -------------------------------------------------
# PROJECT SECTIONS
# -------------------------------------------------

st.subheader("📌 Project Sections")

st.markdown("""
Use the pages in the sidebar to explore:

- 🏠 **Home** – Project overview and objective
- 📊 **Analysis** – Dataset analysis and visualizations
- 🤖 **Model** – Predict ATM failure using the trained Random Forest model
""")


st.markdown("---")


# -------------------------------------------------
# TECHNOLOGIES USED
# -------------------------------------------------

st.subheader("🛠️ Technologies Used")

st.write(
    "Python • Pandas • Scikit-learn • Random Forest • "
    "Streamlit • Power BI"
)
