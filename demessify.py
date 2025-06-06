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

    st.subheader("📊 Raw Data")
    st.dataframe(st.session_state.df)

    if st.button("⬅️ Undo Last Action"):
        undo()

    # Sidebar: preprocessing options
    st.sidebar.header("🧰 Preprocessing Options")
    options = st.sidebar.multiselect(
        "Select preprocessing steps:",
        ["Drop Columns", "Handle Missing Data"]
    )

    # 🔹 Drop Columns
    if "Drop Columns" in options:
        st.subheader("🔹 Drop Columns")
        cols_to_drop = st.multiselect("Select columns to drop:", st.session_state.df.columns.tolist())
        if cols_to_drop:
            if st.button("Apply Drop Columns"):
                push_history()
                st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
                st.success(f"Dropped columns: {', '.join(cols_to_drop)}")

    # 🔹 Handle Missing Data
    if "Handle Missing Data" in options:
        st.subheader("🔹 Handle Missing Data")
        cols_with_na = st.session_state.df.columns[st.session_state.df.isna().any()].tolist()
        if not cols_with_na:
            st.info("No missing data detected.")
        else:
            selected_col = st.selectbox("Select column:", cols_with_na)
            col_dtype = st.session_state.df[selected_col].dtype
            if np.issubdtype(col_dtype, np.number):
                strategy = st.radio("Strategy:", ["Drop Rows", "Fill with Mean", "Fill with Median"])
            else:
                strategy = st.radio("Strategy:", ["Drop Rows", "Fill with Most Frequent"])
            if st.button(f"Apply Missing Data Handling to {selected_col}"):
                push_history()
                if strategy == "Drop Rows":
                    st.session_state.df = st.session_state.df.dropna(subset=[selected_col])
                elif strategy == "Fill with Mean":
                    st.session_state.df[selected_col].fillna(st.session_state.df[selected_col].mean(), inplace=True)
                elif strategy == "Fill with Median":
                    st.session_state.df[selected_col].fillna(st.session_state.df[selected_col].median(), inplace=True)
                elif strategy == "Fill with Most Frequent":
                    st.session_state.df[selected_col].fillna(st.session_state.df[selected_col].mode()[0], inplace=True)
                st.success(f"Applied {strategy} to {selected_col}")

    # 🔎 Processed Data Preview
    st.subheader("🧾 Processed Data Preview")
    st.dataframe(st.session_state.df)

else:
    st.info("Please upload a CSV file to get started.")
