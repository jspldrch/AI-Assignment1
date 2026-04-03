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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

#Settings
INPUT_FILE = "data/processed/final_features.csv"
OUTPUT_DIR = "results/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load Data
df = pd.read_csv(INPUT_FILE)
X = df.drop('label', axis=1)
y = df['label']

#Global Scaling
scaler = StandardScaler()
X_scaled_values = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_values, columns=X.columns)

#1. Learning Curve
def run_data_hunger_experiment():
    
    percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = {"Random Forest": [], "MLP (Deep Learning)": [], "k-NN": []}
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    for p in percentages:
        if p < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train_full, y_train_full, train_size=p, random_state=42, stratify=y_train_full
            )
        else:
            X_sub, y_sub = X_train_full, y_train_full
            
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "MLP (Deep Learning)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
            "k-NN": KNeighborsClassifier(n_neighbors=5)
        }
        
        for name, model in models.items():
            model.fit(X_sub, y_sub)
            results[name].append(accuracy_score(y_test, model.predict(X_test)))

    plt.figure(figsize=(10, 6))
    for model_name, accs in results.items():
        plt.plot(np.array(percentages)*100, accs, marker='o', linewidth=2, label=model_name)
    
    plt.title("Experiment 1: Impact of Training Data Amount", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Training Data used (%)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/01_data_hunger.png")
    plt.close()

#PCA Vizualisation

def run_pca_experiment():
    print("Running Experiment 2: PCA Visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['label'] = y
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='viridis', s=60, alpha=0.8)
    
    var_exp = pca.explained_variance_ratio_
    plt.title(f"Experiment 2: PCA - Feature Space Projection\n(PC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%} variance explained)", fontsize=14, fontweight='bold')
    plt.legend(title="Activities", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_pca_projection.png")
    plt.close()

#Sensor 
def run_sensor_ablation_experiment():
    print("Running Experiment 3: Sensor Importance...")
    
   
    acc_cols = [c for c in X.columns if 'acc' in c.lower()]
    gyro_cols = [c for c in X.columns if 'gyro' in c.lower()]
    
    sensor_sets = {
        "Accelerometer Only": X_scaled[acc_cols],
        "Gyroscope Only": X_scaled[gyro_cols],
        "Full Fusion (Both)": X_scaled
    }
    
    ablation_results = []
    
    for label, data in sensor_sets.items():
        #Use RF 
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf, data, y, cv=5)
        ablation_results.append({"Sensor Configuration": label, "Mean Accuracy": scores.mean()})
    
    res_df = pd.DataFrame(ablation_results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Sensor Configuration", y="Mean Accuracy", palette="magma")
    plt.ylim(0.8, 1.02)
    plt.title("Experiment 3: Sensor Importance Study", fontsize=14, fontweight='bold')
    for i, v in enumerate(res_df["Mean Accuracy"]):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
    
    plt.savefig(f"{OUTPUT_DIR}/03_sensor_ablation.png")
    plt.close()

#Class Imbalance
def run_imbalance_experiment():
    print("Running Experiment 4: Handling Class Imbalance...")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    rf_standard = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    rf_standard.fit(X_train, y_train)
    rf_balanced.fit(X_train, y_train)
    
    f1_std = f1_score(y_test, rf_standard.predict(X_test), average='weighted')
    f1_bal = f1_score(y_test, rf_balanced.predict(X_test), average='weighted')
    
    imb_res = pd.DataFrame({
        "Method": ["Standard Random Forest", "Balanced Random Forest"],
        "Weighted F1-Score": [f1_std, f1_bal]
    })
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=imb_res, x="Method", y="Weighted F1-Score", palette="coolwarm")
    plt.ylim(0.95, 1.01)
    plt.title("Experiment 4: Impact of Class Weighting (Imbalance)", fontsize=14, fontweight='bold')
    for i, v in enumerate(imb_res["Weighted F1-Score"]):
        plt.text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')
    
    plt.savefig(f"{OUTPUT_DIR}/04_imbalance_handling.png")
    plt.close()

#feature reduction 
def run_battery_saving_experiment():
    print("Running Experiment 5: Battery Saving vs. Accuracy...")
    
    #different subsets (6, 12, 36) 
    
    feature_sets = {
        "Low CPU (6 Features)": [c for c in X.columns if 'mean' in c.lower()],
        "Medium CPU (12 Features)": [c for c in X.columns if 'mean' in c.lower() or 'std' in c.lower()],
        "High CPU (36 Features)": X.columns.tolist()
    }
    
    battery_results = []
    
    for label, cols in feature_sets.items():
        data = X_scaled[cols]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf, data, y, cv=5)
        battery_results.append({
            "Complexity Level": label, 
            "Num Features": len(cols),
            "Mean Accuracy": scores.mean()
        })
    
    res_df = pd.DataFrame(battery_results)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x="Num Features", y="Mean Accuracy", marker='s', markersize=10, linewidth=3, color='forestgreen')
    
    plt.title("Experiment 5: Battery Saving (Accuracy vs. CPU Load)", fontsize=14, fontweight='bold')
    plt.xlabel("Computational Load (Number of Features to calculate)", fontsize=12)
    plt.ylabel("Model Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate points
    for i, row in res_df.iterrows():
        plt.text(row["Num Features"], row["Mean Accuracy"] + 0.005, 
                 f"{row['Complexity Level']}\n{row['Mean Accuracy']:.2%}", 
                 ha='center', fontweight='bold', fontsize=10)

    plt.ylim(min(res_df["Mean Accuracy"]) - 0.05, 1.05)
    plt.savefig(f"{OUTPUT_DIR}/05_battery_saving_study.png")
    plt.close()
    print(f"  -> Plot saved: {OUTPUT_DIR}/05_battery_saving_study.png")

if __name__ == "__main__":
    run_data_hunger_experiment()
    run_pca_experiment()
    run_sensor_ablation_experiment()
    run_imbalance_experiment()
    run_battery_saving_experiment()
    print("\nSuccess! All 4 experiments are saved in 'results/experiments/'.")