import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="KNN Trainer", layout="wide")
st.title("📊 Train Your Model with KNN")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Dataset")
    st.dataframe(df.head())

    # Column selection
    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select target column (label)", columns)

    if target_col:
        # Encode categorical features
        for col in df.select_dtypes(include='object').columns:
            df[col] = LabelEncoder().fit_transform(df[col])

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Select K value
        k_value = st.slider("🔢 Select K value for KNN", min_value=1, max_value=20, value=5)

        # Button to train and show accuracy
        if st.button("Train with KNN"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Accuracy of KNN Model: {acc * 100:.2f}%")
