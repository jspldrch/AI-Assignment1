import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. SETUP
INPUT_FILE = "data/processed/final_features.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. LOAD DATA
df = pd.read_csv(INPUT_FILE)
X = df.drop('label', axis=1)
y = df['label']

# 3. SCALING (We scale the whole X for CV, but professionally one should use a Pipeline)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. DEFINE MODELS
models = {
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-Nearest_Neighbors": KNeighborsClassifier(n_neighbors=5),
    "MLP_Deep_Learning": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# 5. CROSS-VALIDATION & EVALUATION
cv_results = {}
final_report = []

# Use StratifiedKFold to keep class balance equal in all folds
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"{'Model':<20} | {'CV Mean Acc':<12} | {'Std Dev':<8}")
print("-" * 45)

for name, model in models.items():
    # Choose scaled data for k-NN and MLP, raw data for RF
    current_X = X_scaled if name != "Random_Forest" else X
    
    # Perform 5-Fold Cross-Validation
    scores = cross_val_score(model, current_X, y, cv=skf)
    
    mean_acc = scores.mean()
    std_acc = scores.std()
    cv_results[name] = mean_acc
    
    print(f"{name:<20} | {mean_acc:.4f}      | {std_acc:.4f}")

    # --- Standard Train-Test Split for Confusion Matrix & Report ---
    # (CV gives you the score, but for the Matrix we still need one specific split)
    X_train, X_test, y_train, y_test = train_test_split(current_X, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save Report
    with open(f"{RESULTS_DIR}/report_{name}.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Cross-Validation Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n\n")
        f.write(classification_report(y_test, y_pred))

    # Save Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Confusion Matrix: {name}\nCV Accuracy: {mean_acc:.2%}")
    plt.savefig(f"{RESULTS_DIR}/cm_{name}.png")
    plt.close()

# 6. SAVE COMPARISON PLOT
plt.figure(figsize=(10,6))
plt.bar(cv_results.keys(), cv_results.values(), color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylabel('Mean CV Accuracy')
plt.title('5-Fold Cross-Validation Performance')
plt.ylim(0, 1.1)
for i, v in enumerate(cv_results.values()):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
plt.savefig(f"{RESULTS_DIR}/cv_accuracy_comparison.png")

print(f"\nDone! Results with Cross-Validation are in '{RESULTS_DIR}'.")