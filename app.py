import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="ML Classifier or Regressor", layout="wide")
st.title("🤖1 Auto ML: Classification or Regression Model Trainer")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV format)", type=["csv"])

# Detect classification or regression
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

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode features
        X = pd.get_dummies(X)

        # Encode target if classification and not numeric
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("⚙️ Choose Model")

        if problem_type == "classification":
            knn_clicked = st.button("📌 Calculate KNN Classifier Accuracy")
            dt_clicked = st.button("🌳 Calculate Decision Tree Classifier Accuracy")
            logreg_clicked = st.button("📈 Calculate Logistic Regression Accuracy")
            svm_clicked = st.button("📊 Calculate SVM Classifier Accuracy")
            rf_clicked = st.button("🌲 Calculate Random Forest Classifier Accuracy")

            if knn_clicked:
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ KNN Classifier Accuracy: {acc:.2f}")

            if dt_clicked:
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Decision Tree Classifier Accuracy: {acc:.2f}")

            if logreg_clicked:
                lr = LogisticRegression(max_iter=1000)
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Logistic Regression Accuracy: {acc:.2f}")

            if svm_clicked:
                svm = SVC()
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ SVM Classifier Accuracy: {acc:.2f}")

            if rf_clicked:
                rf = RandomForestClassifier(random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Random Forest Classifier Accuracy: {acc:.2f}")

        elif problem_type == "regression":
            knn_reg_clicked = st.button("📌 Calculate KNN Regression Metrics")
            dt_reg_clicked = st.button("🌳 Calculate Decision Tree Regression Metrics")
            linreg_clicked = st.button("📈 Calculate Linear Regression Metrics")
            svr_clicked = st.button("📊 Calculate SVR Metrics")
            rf_reg_clicked = st.button("🌲 Calculate Random Forest Regression Metrics")

            if knn_reg_clicked:
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                st.success("✅ KNN Regressor Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")

            if dt_reg_clicked:
                dt = DecisionTreeRegressor(random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                st.success("✅ Decision Tree Regressor Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")

            if linreg_clicked:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)
                st.success("✅ Linear Regression Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")

            if svr_clicked:
                svr = SVR()
                svr.fit(X_train, y_train)
                y_pred = svr.predict(X_test)
                st.success("✅ SVR Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")

            if rf_reg_clicked:
                rf = RandomForestRegressor(random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                st.success("✅ Random Forest Regressor Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
