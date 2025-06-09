import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

st.set_page_config(page_title="DeMessify", layout="wide")
st.title("🧹 DeMessify - Turn data dirt into gold")

# Initialize session state
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
        [
            "Drop Columns",
            "Handle Missing Data",
            "Handle Outliers",
            "Drop Duplicates",
            "Encode Categorical Variables",
            "Feature Scaling/Normalization",
            "Feature Engineering"
        ]
    )

    # 🔹 Drop Columns
    if "Drop Columns" in options:
        st.subheader("🔹 Drop Columns")
        cols_to_drop = st.multiselect("Select columns to drop:", st.session_state.df.columns.tolist())
        if cols_to_drop and st.button("Apply Drop Columns"):
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

    # 🔹 Handle Outliers
    if "Handle Outliers" in options:
        st.subheader("🔹 Handle Outliers")
        method = st.selectbox("Method:", ["Z-Score", "IQR"])
        num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected_cols = st.multiselect("Select columns:", num_cols)
            if selected_cols and st.button("Apply Outlier Removal"):
                push_history()
                original_size = len(st.session_state.df)
                if method == "Z-Score":
                    z_scores = st.session_state.df[selected_cols].apply(zscore)
                    st.session_state.df = st.session_state.df[(z_scores.abs() < 3).all(axis=1)]
                elif method == "IQR":
                    for col in selected_cols:
                        Q1 = st.session_state.df[col].quantile(0.25)
                        Q3 = st.session_state.df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        st.session_state.df = st.session_state.df[
                            (st.session_state.df[col] >= Q1 - 1.5 * IQR) &
                            (st.session_state.df[col] <= Q3 + 1.5 * IQR)
                        ]
                removed = original_size - len(st.session_state.df)
                st.success(f"Removed {removed} outliers using {method} method")

    # 🔹 Drop Duplicates
    if "Drop Duplicates" in options:
        st.subheader("🔹 Remove Duplicates")
        if st.button("Remove Duplicates"):
            push_history()
            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates()
            after = len(st.session_state.df)
            st.success(f"Removed {before - after} duplicate rows")

    # 🔹 Encode Categorical Variables
    if "Encode Categorical Variables" in options:
        st.subheader("🔹 Encode Categorical Variables")
        cat_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            selected_cols = st.multiselect("Select categorical columns:", cat_cols)
            encoding_method = st.radio("Encoding Method:", ["Label Encoding", "One-Hot Encoding", "Get Dummies"])
            if selected_cols and st.button("Apply Encoding"):
                push_history()
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    for col in selected_cols:
                        st.session_state.df[col] = le.fit_transform(st.session_state.df[col].astype(str))
                elif encoding_method == "One-Hot Encoding":
                    df_encoded = pd.get_dummies(st.session_state.df[selected_cols], drop_first=False)
                    st.session_state.df = pd.concat(
                        [st.session_state.df.drop(columns=selected_cols), df_encoded], axis=1
                    )
                else:  # Get Dummies
                    st.session_state.df = pd.get_dummies(
                        st.session_state.df, columns=selected_cols, drop_first=True
                    )
                st.success(f"Encoding applied using {encoding_method}")
        else:
            st.info("No categorical columns found")

    # 🔹 Feature Scaling / Normalization
    if "Feature Scaling/Normalization" in options:
        st.subheader("🔹 Feature Scaling")
        num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected_cols = st.multiselect("Select numeric columns:", num_cols)
            scaling_method = st.radio("Scaling Method:", ["StandardScaler", "MinMaxScaler"])
            if selected_cols and st.button("Apply Scaling"):
                push_history()
                scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
                st.session_state.df[selected_cols] = scaler.fit_transform(st.session_state.df[selected_cols])
                st.success(f"{scaling_method} applied successfully")
        else:
            st.warning("No numeric columns for scaling")

    # 🔹 Feature Engineering
    if "Feature Engineering" in options:
        st.subheader("🔹 Feature Engineering")
        converted_cols = []
        for col in st.session_state.df.columns:
            if st.session_state.df[col].dtype == 'object':
                try:
                    st.session_state.df[col] = pd.to_datetime(st.session_state.df[col])
                    converted_cols.append(col)
                except (ValueError, TypeError):
                    pass
        if converted_cols:
            st.info(f"Automatically converted to datetime: {', '.join(converted_cols)}")

        date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
        if date_cols:
            selected_cols = st.multiselect("Select datetime columns to extract features from:", date_cols)
            if selected_cols and st.button("Extract Year, Month, Day"):
                push_history()
                for col in selected_cols:
                    st.session_state.df[f"{col}_year"] = st.session_state.df[col].dt.year
                    st.session_state.df[f"{col}_month"] = st.session_state.df[col].dt.month
                    st.session_state.df[f"{col}_day"] = st.session_state.df[col].dt.day
                st.success("Date features extracted")
        else:
            st.info("No datetime columns found")

    # 🧾 Final Processed Data + Download
    st.subheader("🧾 Processed Data Preview")
    st.dataframe(st.session_state.df)

    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Cleaned Data",
        csv,
        "cleaned_data.csv",
        "text/csv",
        key='download-csv'
    )

else:
    st.info("Please upload a CSV file to get started.")
