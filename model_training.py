# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/windowed_features.csv"
MODEL_PATH = "driver_behavior_model.pkl"

# Load cleaned data
df = pd.read_csv(DATA_PATH)
label_column = "Target"
X = df.drop(columns=[label_column])
y = df[label_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and feature names
joblib.dump((model, X.columns.tolist()), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# === Feature importance visualization ===
importances = model.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices][:20], y=X.columns[indices][:20])
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.show()
print("Feature importance saved to feature_importance.png")

