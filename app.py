import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="ML Model Type Detector", layout="wide")
st.title("📊 Auto ML: Classification or Regression")

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])

def detect_problem_type(df, target_col):
    unique_vals = df[target_col].nunique()
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return "classification"
    elif unique_vals <= 10:
        return "classification"
    else:
        return "regression"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if target_col:
        # Detect problem type
        problem_type = detect_problem_type(df, target_col)
        st.info(f"🧠 Detected Problem Type: **{problem_type.upper()}**")

        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode non-numeric features in X
        X = pd.get_dummies(X)

        # Encode target if classification and categorical
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("⚙️ Choose Model")

        if problem_type == "classification":
            if st.button("📌 Calculate KNN Accuracy"):
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ KNN Classifier Accuracy: {acc:.2f}")

            if st.button("🌳 Calculate Decision Tree Accuracy"):
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Decision Tree Classifier Accuracy: {acc:.2f}")

        elif problem_type == "regression":
            if st.button("📌 Calculate KNN Regression Metrics"):
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                st.success("✅ KNN Regressor Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R²: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")

            if st.button("🌳 Calculate Decision Tree Regression Metrics"):
                dt = DecisionTreeRegressor(random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                st.success("✅ Decision Tree Regressor Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R²: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
