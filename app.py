import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

# Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

st.set_page_config(page_title="ML Classifier or Regressor", layout="wide")
st.title("🤖 Auto ML: Classification or Regression Model Trainer")

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV format)", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def detect_problem_type(df, target_col):
    unique_vals = df[target_col].nunique()
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return "classification"
    elif unique_vals <= 10:
        return "classification"
    else:
        return "regression"

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if target_col:
        problem_type = detect_problem_type(df, target_col)
        st.info(f"🧠 Detected Problem Type: **{problem_type.upper()}**")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X)

        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("⚙️ Choose Model")

        if problem_type == "classification":
            models = {
                "KNN Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Random Forest Classifier": RandomForestClassifier(),
                "SVM Classifier": SVC(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Naive Bayes": GaussianNB(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                "MLP Classifier": MLPClassifier(max_iter=300),
                "Bagging Classifier": BaggingClassifier(),
                "Extra Trees Classifier": ExtraTreesClassifier(),
            }

            selected_model = st.selectbox("Choose Classification Model", list(models.keys()))
            if st.button("Train Selected Classification Model"):
                model = models[selected_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ {selected_model} Accuracy: {acc:.2f}")

        elif problem_type == "regression":
            models = {
                "KNN Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "SVR": SVR(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet Regression": ElasticNet(),
                "Bayesian Ridge Regression": BayesianRidge(),
                "MLP Regressor": MLPRegressor(max_iter=300),
                "Bagging Regressor": BaggingRegressor(),
                "Extra Trees Regressor": ExtraTreesRegressor(),
            }

            selected_model = st.selectbox("Choose Regression Model", list(models.keys()))
            if st.button("Train Selected Regression Model"):
                model = models[selected_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.success(f"✅ {selected_model} Results")
                st.write(f"📉 MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
