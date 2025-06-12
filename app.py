import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="KNN Model Trainer", layout="wide")
st.title("📊 KNN Machine Learning Trainer")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Step 2: Select target column
    all_columns = df.columns.tolist()
    target_col = st.selectbox("Select the target column (label)", all_columns)

    # Step 3: Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Choose K value
    k_value = st.slider("Select K value for KNN", min_value=1, max_value=20, value=5)

    # Step 6: Train and evaluate
    if st.button("Train KNN and Show Accuracy"):
        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Accuracy: {acc * 100:.2f}%")
