import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Classifier", layout="wide")
st.title("📊 ML Model Trainer (KNN + Decision Tree)")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select target column", columns)

    if target_col:
        # Show label distribution
        st.write("📈 Target label distribution:")
        st.dataframe(df[target_col].value_counts())

        # Check for classification suitability
        if df[target_col].dtype in ['float64', 'float32'] and df[target_col].nunique() > 10:
            st.error("❌ Target column seems continuous. Use regression instead.")
        else:
            # Encode only categorical features (not target)
            df_encoded = df.copy()
            for col in df_encoded.columns:
                if col != target_col and df_encoded[col].dtype == 'object':
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

            X = df_encoded.drop(target_col, axis=1)
            y = df[target_col]  # do NOT encode the target

            # Encode target if it's object
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # UI buttons
            if "run_knn" not in st.session_state:
                st.session_state.run_knn = False
            if "run_dt" not in st.session_state:
                st.session_state.run_dt = False

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Train with KNN"):
                    st.session_state.run_knn = True
                    st.session_state.run_dt = False

            with col2:
                if st.button("🌳 Train with Decision Tree"):
                    st.session_state.run_dt = True
                    st.session_state.run_knn = False

            if st.session_state.run_knn:
                k = st.slider("🔢 Select K value", 1, 20, 5)
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"KNN Accuracy (K={k}): {acc * 100:.2f}%")

            if st.session_state.run_dt:
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Decision Tree Accuracy: {acc * 100:.2f}%")
