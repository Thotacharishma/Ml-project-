import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ML Regression Trainer", layout="wide")
st.title("📈 ML Model Trainer (KNN & Decision Tree for Regression)")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select target column (must be numeric/continuous)", columns)

    if target_col:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            st.error("❌ Please select a numeric column for regression.")
        else:
            # Encode categorical features (if any)
            df_encoded = df.copy()
            for col in df_encoded.columns:
                if col != target_col and df_encoded[col].dtype == 'object':
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

            X = df_encoded.drop(target_col, axis=1)
            y = df_encoded[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Buttons to choose model
            col1, col2 = st.columns(2)
            if "run_knn" not in st.session_state:
                st.session_state.run_knn = False
            if "run_dt" not in st.session_state:
                st.session_state.run_dt = False

            with col1:
                if st.button("🔍 Train with KNN Regressor"):
                    st.session_state.run_knn = True
                    st.session_state.run_dt = False

            with col2:
                if st.button("🌳 Train with Decision Tree Regressor"):
                    st.session_state.run_dt = True
                    st.session_state.run_knn = False

            if st.session_state.run_knn:
                k = st.slider("🔢 Select K value", 1, 20, 5)
                knn = KNeighborsRegressor(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.success(f"✅ KNN Regressor (K={k})\n📉 MSE: {mse:.2f}\n📈 R² Score: {r2:.2f}")

            if st.session_state.run_dt:
                dt = DecisionTreeRegressor(random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.success(f"✅ Decision Tree Regressor\n📉 MSE: {mse:.2f}\n📈 R² Score: {r2:.2f}")
