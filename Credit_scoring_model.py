import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Create Synthetic Dataset
np.random.seed(42)
n_samples = 200

data = {
    "Income": np.random.randint(20000, 100000, n_samples),
    "Debt": np.random.randint(0, 50000, n_samples),
    "Years_of_Credit_History": np.random.randint(1, 30, n_samples),
    "Number_of_Late_Payments": np.random.randint(0, 10, n_samples),
    "Credit_Utilization_Ratio": np.random.uniform(0, 1, n_samples),
    "Creditworthy": np.random.randint(0, 2, n_samples)  # Target (0 = No, 1 = Yes)
}

df = pd.DataFrame(data)

# Step 2: Separate Features and Target
X = df.drop("Creditworthy", axis=1)
y = df["Creditworthy"]

# Step 3: Split into Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Initialize Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Step 6: Train & Evaluate Models
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
