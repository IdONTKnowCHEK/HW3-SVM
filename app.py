import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

st.title("Logistic Regression and SVM Classifier on Iris and Breast Cancer Datasets")

# Load Datasets
iris = load_iris()
breast_cancer = load_breast_cancer()

# Sidebar - Dataset selection
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer"))

# Sidebar - Model parameters
st.sidebar.header("Model Parameters")
if dataset_name == "Iris":
    X, y = pd.DataFrame(data=iris.data, columns=iris.feature_names), iris.target
    feature_names = iris.feature_names
else:
    X, y = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names), breast_cancer.target
    feature_names = breast_cancer.feature_names

# Model selection
model_name = st.sidebar.selectbox("Select Model", ("Logistic Regression", "SVM"))

# Parameters for Logistic Regression
if model_name == "Logistic Regression":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, step=0.01)
    max_iter = st.sidebar.slider("Max Iterations", 100, 500, step=50)
    model = LogisticRegression(C=C, max_iter=max_iter)

# Parameters for SVM
else:
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, step=0.01)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly", "sigmoid"))
    model = SVC(C=C, kernel=kernel)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :2], y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.write(f"### {model_name} Results on {dataset_name} Dataset")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Display
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
st.pyplot(fig)

# Plot decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap='coolwarm', marker='o', label="Train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', cmap='coolwarm', marker='s', label="Test")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title(f"{model_name} Decision Boundary on {dataset_name} Dataset")
plt.legend()
st.pyplot(plt.gcf())
