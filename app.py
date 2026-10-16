from pathlib import Path

import streamlit as st

# Configure the Streamlit app once, in the entry point.
st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

revenue_page = st.Page(
    str(BASE_DIR / "pages" / "revenue_prediction.py"),
    title="Revenue Prediction",
    icon="📈",
    default=True,
)

feedback_page = st.Page(
    str(BASE_DIR / "2_Customer_Feedback_Analysis.py"),
    title="Customer Feedback Analysis",
    icon="💬",
)

navigation = st.navigation([revenue_page, feedback_page])
navigation.run()
