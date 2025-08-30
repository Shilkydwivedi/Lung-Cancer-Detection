import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Encode target column
label_enc = LabelEncoder()
df['Lung_Cancer'] = label_enc.fit_transform(df['Lung_Cancer'])  # YES=1, NO=0

# Features (X) and Target (y)
X = df.drop('Lung_Cancer', axis=1)
y = df['Lung_Cancer']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\n✅ Model Training Completed!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
