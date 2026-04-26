import os
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.express as px


def detect_problem_type(df, target):
    if df[target].nunique() < 10:
        return "classification"
    return "regression"


def auto_train_models(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    problem = detect_problem_type(df, target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results = {}
    trained_models = {}

    if problem == "classification":

        models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            results[name] = acc
            trained_models[name] = model

    else:

        models = {
            "RandomForest": RandomForestRegressor(),
            "LinearRegression": LinearRegression()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            results[name] = mse
            trained_models[name] = model

    # 🔹 Show Results
    st.subheader("📊 Model Comparison")

    for model, score in results.items():
        st.write(f"{model}: {score}")

    # 🔹 Select Best Model
    best_model_name = (
        max(results, key=results.get)
        if problem == "classification"
        else min(results, key=results.get)
    )

    best_model = trained_models[best_model_name]

    st.success(f"🏆 Best Model: {best_model_name}")

    # 🔥 Feature Importance (only for RandomForest)
    if "RandomForest" in trained_models:
        rf_model = trained_models["RandomForest"]

        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            feature_names = X.columns

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.subheader("🔥 Feature Importance")

            fig = px.bar(
                importance_df,
                x="Feature",
                y="Importance"
            )

            st.plotly_chart(fig, use_container_width=True)

    # 🔹 Save Model
    os.makedirs("outputs/models", exist_ok=True)
    model_path = f"outputs/models/{best_model_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # 🔹 Download Button
    with open(model_path, "rb") as f:
        st.download_button(
            label="💾 Download Best Model",
            data=f,
            file_name=f"{best_model_name}.pkl"
        )