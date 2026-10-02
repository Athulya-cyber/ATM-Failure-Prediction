import streamlit as st


st.set_page_config(
    page_title="ATM Failure Prediction",
    page_icon="🏧",
    layout="wide")



pages =  {
    "Main" : [
        st.Page("pages/home.py", title="Home", icon="🏠"),
        st.Page("pages/analysis.py", title="Data Analysis", icon="📊"),
        st.Page("pages/model.py", title="Prediction", icon="🔮")]}

pg = st.navigation(pages)

pg.run()
