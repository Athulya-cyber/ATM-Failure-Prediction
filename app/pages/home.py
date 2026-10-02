import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

st.title("🏧 ATM Failure Prediction")

st.subheader("Machine Learning Project")

st.write("This application uses a Random Forest Classifier to predict whether an ATM will fail in the next 48 hours or not.")

st.markdown("---")


st.subheader("Project Overview")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ATM_Maintenance_Cleaned.csv"

atm = pd.read_csv(DATA_PATH)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        atm.shape[0])

with col2:
    st.metric(
        "Features Used",
        atm.shape[1])

with col3:
    st.metric(
        "Machine Learning Model",
        "Random Forest")

st.markdown("---")


st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "100%")

with col2:
    st.metric("Precision", "100%")

with col3:
    st.metric("Recall", "100%")

with col4:
    st.metric("F1 Score", "100%")

st.markdown("---")


st.markdown("""
### Project Sections

Use the pages in the sidebar to explore:

- 🏠 **Home** – Problem Statement and Project Overview
- 📊 **Analysis** – Dataset Analysis and Visualizations
- 🤖 **Model** – Model Prediction
""")
