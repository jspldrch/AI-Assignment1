import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Settings
INPUT_FILE = "data/processed/final_features.csv"
OUTPUT_DIR = "results/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
df = pd.read_csv(INPUT_FILE)
X = df.drop('label', axis=1)
y = df['label']

# Global Scaling (important for PCA and MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Data Hunger (Learning Curves)
# ─────────────────────────────────────────────────────────────────
def run_data_hunger_experiment():
    print("Running Experiment 1: Data Hunger...")
    
    percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = {"Random_Forest": [], "MLP_Deep_Learning": [], "k-NN": []}
    
    # We split off a fixed 20% test set to compare everything fairly
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    for p in percentages:
        # Take a subset of the training data
        if p < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train_full, y_train_full, train_size=p, random_state=42, stratify=y_train_full
            )
        else:
            X_sub, y_sub = X_train_full, y_train_full
            
        # Train Models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        
        rf.fit(X_sub, y_sub)
        mlp.fit(X_sub, y_sub)
        knn.fit(X_sub, y_sub)
        
        results["Random_Forest"].append(accuracy_score(y_test, rf.predict(X_test)))
        results["MLP_Deep_Learning"].append(accuracy_score(y_test, mlp.predict(X_test)))
        results["k-NN"].append(accuracy_score(y_test, knn.predict(X_test)))

    # Plotting
    plt.figure(figsize=(10, 6))
    for model_name, accs in results.items():
        plt.plot(np.array(percentages)*100, accs, marker='o', label=model_name)
    
    plt.title("Experiment 1: Impact of Training Data Amount")
    plt.xlabel("Percentage of Training Data used (%)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/experiment_data_hunger.png")
    plt.close()
    print(f"  -> Plot saved: {OUTPUT_DIR}/experiment_data_hunger.png")

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Dimensionality Reduction (PCA Visualization)
# ─────────────────────────────────────────────────────────────────
def run_pca_experiment():
    print("Running Experiment 2: PCA Visualization...")
    
    # Reduce 36 features to 2 Principal Components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['label'] = y
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='viridis', alpha=0.7)
    
    # Explanation for report: Explained Variance
    var_exp = pca.explained_variance_ratio_
    plt.title(f"Experiment 2: PCA - Feature Space Projection\n(PC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%} variance explained)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/experiment_pca_projection.png")
    plt.close()
    
    # Calculate Accuracy with only 2 PCA components vs All Features
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    rf_pca = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    acc_pca = accuracy_score(y_test, rf_pca.predict(X_test))
    
    print(f"  -> Plot saved: {OUTPUT_DIR}/experiment_pca_projection.png")
    print(f"  -> Accuracy with only 2 PCA components: {acc_pca:.2%}")

if __name__ == "__main__":
    run_data_hunger_experiment()
    run_pca_experiment()
    print("\nAll experiments completed successfully!")
