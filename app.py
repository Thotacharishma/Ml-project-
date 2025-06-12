import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="KNN Trainer", layout="wide")
st.title("📊 Train Model using KNN")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Let user pick the target column
    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select the target column", columns)

    if target_col:
        # Slider to choose K
        k_value = st.slider("🔢 Choose value of K", min_value=1, max_value=20, value=5)

        # Show KNN button after data is uploaded and target is selected
        if st.button("🚀 Run KNN"):
            # Label encode if needed
            for col in df.select_dtypes(include='object').columns:
                df[col] = LabelEncoder().fit_transform(df[col])

            X = df.drop(target_col, axis=1)
            y = df[target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # KNN model
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Accuracy score
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ KNN Model Accuracy: {acc * 100:.2f}%")
