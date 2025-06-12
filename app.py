import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Trainer", layout="wide")
st.title("🧠 Train a Classification Model (KNN or Decision Tree)")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Target column selection
    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select the target column", columns)

    if target_col:
        # Warn if target column is continuous
        if df[target_col].dtype in ['float64', 'float32']:
            st.warning("⚠️ Selected target column seems continuous. KNN and Decision Tree here support only classification.")
        else:
            # Encode categorical features
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include='object').columns:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

            X = df_encoded.drop(target_col, axis=1)
            y = df_encoded[target_col]

            # Train-Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Button logic
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

            # Train KNN
            if st.session_state.run_knn:
                k = st.slider("🔢 Select value of K", 1, 20, 5)
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ KNN Accuracy (K={k}): {acc * 100:.2f}%")

            # Train Decision Tree
            if st.session_state.run_dt:
                dtree = DecisionTreeClassifier(random_state=42)
                dtree.fit(X_train, y_train)
                y_pred = dtree.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Decision Tree Accuracy: {acc * 100:.2f}%")
