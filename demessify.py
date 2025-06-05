import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DeMessify", layout="wide")
st.title("🧹 DeMessify - Turn data dirt into gold")

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'history' not in st.session_state:
    st.session_state.history = []

def push_history():
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())

def undo():
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()
        st.success("Undid last action")
    else:
        st.warning("No actions to undo")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.history = []

    st.subheader("Raw Data")
    st.dataframe(st.session_state.df)

    if st.button("⬅️ Undo Last Action"):
        undo()
else:
    st.info("Please upload a CSV file to get started.")
