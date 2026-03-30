import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# 1. SETUP & DIRECTORIES
# ─────────────────────────────────────────────
DATA_PATH = "data/processed/final_features.csv"
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Features (X) and Labels (y)
X = df.drop('label', axis=1)
y = df['label']
class_names = sorted(y.unique())

# Split into Training (70%) and Testing (30%)
# 'stratify=y' ensures each activity is equally represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scaling (Mandatory for k-NN and MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. DEFINE MODELS
# ─────────────────────────────────────────────
models = {
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-Nearest_Neighbors": KNeighborsClassifier(n_neighbors=5),
    "MLP_Deep_Learning": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION LOOP
# ─────────────────────────────────────────────
accuracy_comparison = {}

for name, model in models.items():
    print(f"Currently training: {name}...")
    
    # Use scaled data for k-NN and MLP; RF works fine with both
    X_tr = X_train_scaled if name != "Random_Forest" else X_train
    X_te = X_test_scaled if name != "Random_Forest" else X_test
    
    # Train
    model.fit(X_tr, y_train)
    
    # Predict
    y_pred = model.predict(X_te)
    
    # Calculate Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracy_comparison[name] = acc
    
    # 5. SAVE CLASSIFICATION REPORT (Text)
    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_FOLDER, f"report_{name}.txt"), "w") as f:
        f.write(f"Model: {name}\nAccuracy: {acc:.4f}\n\n")
        f.write(report)
    
    # 6. SAVE CONFUSION MATRIX (Graph)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {name}\nAccuracy: {acc:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"cm_{name}.png"))
    plt.close()

# ─────────────────────────────────────────────
# 5. FINAL COMPARISON PLOT
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
bars = plt.bar(accuracy_comparison.keys(), accuracy_comparison.values(), color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylim(0, 1.1)
plt.ylabel('Accuracy Score')
plt.title('Performance Comparison: Accuracy')

# Add percentage labels on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "accuracy_comparison.png"))
plt.close()

print(f"\nDone! Please check the folder: '{OUTPUT_FOLDER}' for all reports and graphs.")
