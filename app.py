import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Trainer", layout="wide")
st.title("🧠 Train a Model (KNN or Decision Tree)")

# Upload dataset
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Target column selection
    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select the target column", columns)

    if target_col:
        # Encode categorical features
        for col in df.select_dtypes(include='object').columns:
            df[col] = LabelEncoder().fit_transform(df[col])

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Buttons to train
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Train with KNN"):
                # Show slider *inside* the button block
                k_value = st.slider("🔢 Select value of K for KNN", 1, 20, 5)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                knn = KNeighborsClassifier(n_neighbors=k_value)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"KNN Model Accuracy (K={k_value}): {acc * 100:.2f}%")

        with col2:
            if st.button("🌳 Train with Decision Tree"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                dtree = DecisionTreeClassifier(random_state=42)
                dtree.fit(X_train, y_train)
                y_pred = dtree.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Decision Tree Accuracy: {acc * 100:.2f}%")
