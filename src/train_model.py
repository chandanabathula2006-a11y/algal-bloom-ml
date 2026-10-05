import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
df = pd.read_csv("data/bloom_dataset.csv")

# Features used by the ML model
features = [
    "Temperature",
    "Specific_Conductivity",
    "DO_Saturation",
    "Dissolved_Oxygen",
    "Chlorophyll",
    "Turbidity",
    "Phycocyanin",
    "pH"
]

# Target
target = "Bloom_Risk_Label"

X = df[features]
y = df[target]

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ML pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# Train
print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Training completed.")

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
rf_model = pipeline.named_steps["model"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Save model
joblib.dump(pipeline, "models/bloom_model.pkl")

# Save feature list
with open("models/features.json", "w") as f:
    json.dump(features, f, indent=4)

print("\nModel saved to: models/bloom_model.pkl")
print("Feature list saved to: models/features.json")