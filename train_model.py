import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# -----------------------------
# Generate Healthcare Dataset
# -----------------------------

np.random.seed(42)

# Age, BMI, Blood Pressure, Glucose,
# Heart Rate, Cholesterol
X = np.random.randint(18, 80, (1000, 6))

age = X[:, 0]
bmi = X[:, 1]
bp = X[:, 2]
glucose = X[:, 3]
heart_rate = X[:, 4]
cholesterol = X[:, 5]


# -----------------------------
# Calculate Risk Score
# -----------------------------

risk_score = (
    0.20 * ((age - 18) / 62) +
    0.15 * np.clip((bmi - 18) / 22, 0, 1) +
    0.25 * np.clip((bp - 90) / 90, 0, 1) +
    0.25 * np.clip((glucose - 70) / 130, 0, 1) +
    0.05 * np.clip((heart_rate - 60) / 80, 0, 1) +
    0.10 * np.clip((cholesterol - 120) / 180, 0, 1)
)


# Add small random variation
risk_score += np.random.normal(0, 0.08, len(X))

risk_score = np.clip(risk_score, 0, 1)


# Convert probability into classes
y = (risk_score >= np.median(risk_score)).astype(int)


# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Decision Tree
# -----------------------------

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

dt.fit(X_train, y_train)


# -----------------------------
# Logistic Regression
# -----------------------------

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000
    )
)

lr.fit(X_train, y_train)


# -----------------------------
# Accuracy
# -----------------------------

dt_accuracy = accuracy_score(
    y_test,
    dt.predict(X_test)
)

lr_accuracy = accuracy_score(
    y_test,
    lr.predict(X_test)
)

print(
    "Decision Tree Accuracy:",
    round(dt_accuracy * 100, 2),
    "%"
)

print(
    "Logistic Regression Accuracy:",
    round(lr_accuracy * 100, 2),
    "%"
)


# -----------------------------
# Save Models
# -----------------------------

joblib.dump(
    dt,
    "decision_tree.pkl"
)

joblib.dump(
    lr,
    "logistic_regression.pkl"
)

print("Models saved successfully!")


# -----------------------------
# Accuracy Chart
# -----------------------------

models = [
    "Decision Tree",
    "Logistic Regression"
]

accuracies = [
    dt_accuracy,
    lr_accuracy
]


plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    accuracies
)

plt.ylim(0, 1.1)

plt.ylabel("Accuracy")

plt.xlabel("Machine Learning Model")

plt.title("Model Accuracy Comparison")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


# Accuracy labels

for bar, accuracy in zip(
    bars,
    accuracies
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,

        accuracy + 0.02,

        f"{accuracy * 100:.1f}%",

        ha="center",

        fontweight="bold"
    )


plt.tight_layout()


# Save chart directly to static

plt.savefig(
    "static/model_accuracy.png",
    dpi=150
)

plt.close()

print("Accuracy chart created!")