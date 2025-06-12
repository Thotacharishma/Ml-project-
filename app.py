import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ML Regressor", layout="wide")
st.title("📈 Train Regression Models (KNN & Decision Tree)")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Select target
    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select Target Column (must be numeric)", columns)

    if target_col:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            st.error("❌ Target must be numeric for regression.")
        else:
            # Encode non-numeric features
            df_encoded = df.copy()
            for col in df_encoded.columns:
                if col != target_col and df_encoded[col].dtype == 'object':
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

            X = df_encoded.drop(target_col, axis=1)
            y = df_encoded[target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Button logic
            col1, col2 = st.columns(2)
            model_type = st.radio("🤖 Choose Model", ["KNN Regressor", "Decision Tree Regressor"])

            if model_type == "KNN Regressor":
                k = st.slider("🔢 Choose value of K", 1, 20, 5)
                if st.button("🚀 Train KNN Regressor"):
                    model = KNeighborsRegressor(n_neighbors=k)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.success(f"✅ KNN Regressor Results (K={k})")
                    st.write(f"📉 **MSE**: {mse:.2f}")
                    st.write(f"📈 **R² Score**: {r2:.2f}")

            elif model_type == "Decision Tree Regressor":
                if st.button("🚀 Train Decision Tree Regressor"):
                    model = DecisionTreeRegressor(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.success("✅ Decision Tree Regressor Results")
                    st.write(f"📉 **MSE**: {mse:.2f}")
                    st.write(f"📈 **R² Score**: {r2:.2f}")
