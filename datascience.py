import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

st.title("Model Performance Comparison")

# Sidebar options
st.sidebar.header("Model Settings")
n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 5, 50, 20)
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)

# Generate dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

# Fit models and compute metrics
metrics = {name:{} for name in models.keys()}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_scores = model.decision_function(X_test_scaled).reshape(-1,1)
        y_prob = MinMaxScaler().fit_transform(y_scores).ravel()

    metrics[name]["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics[name]["Precision"] = precision_score(y_test, y_pred)
    metrics[name]["Recall"] = recall_score(y_test, y_pred)
    metrics[name]["F1-score"] = f1_score(y_test, y_pred)
    metrics[name]["ROC-AUC"] = roc_auc_score(y_test, y_prob)

# Show metrics in a table
st.subheader("Metrics Table")
st.table(metrics)

# Plot metrics
st.subheader("Metrics Comparison Chart")
metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
x = np.arange(len(metric_names))
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))
for i, (model_name, scores) in enumerate(metrics.items()):
    values = [scores[m] for m in metric_names]
    ax.bar(x + i*width, values, width, label=model_name)

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metric_names)
ax.legend()
plt.ylim(0,1)
st.pyplot(fig)
