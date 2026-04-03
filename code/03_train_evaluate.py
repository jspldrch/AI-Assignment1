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

def save_table_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    #table
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     rowLabels=df.index if df.index.name != None or not isinstance(df.index, pd.RangeIndex) else None,
                     cellLoc='center', 
                     loc='center')
    
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5) 
    
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



INPUT_FILE = "data/processed/final_features.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#load data
df = pd.read_csv(INPUT_FILE)
X = df.drop('label', axis=1)
y = df['label']

#scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#model setup
models = {
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-Nearest_Neighbors": KNeighborsClassifier(n_neighbors=5),
    "MLP_Deep_Learning": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

#validation
cv_results = {}
final_report = []


#stratified
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"{'Model':<20} | {'CV Mean Acc':<12} | {'Std Dev':<8}")
print("-" * 45)

for name, model in models.items():
    
    current_X = X_scaled if name != "Random_Forest" else X
    
   
    scores = cross_val_score(model, current_X, y, cv=skf)
    
    mean_acc = scores.mean()
    std_acc = scores.std()
    cv_results[name] = mean_acc
    
    print(f"{name:<20} | {mean_acc:.4f}      | {std_acc:.4f}")

    #Train-Test Split
    
    X_train, X_test, y_train, y_test = train_test_split(current_X, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #report
    with open(f"{RESULTS_DIR}/report_{name}.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Cross-Validation Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n\n")
        f.write(classification_report(y_test, y_pred))

    #Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Confusion Matrix: {name}\nCV Accuracy: {mean_acc:.2%}")
    plt.savefig(f"{RESULTS_DIR}/cm_{name}.png")
    plt.close()

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    save_table_as_image(report_df, f"{RESULTS_DIR}/table_img_{name}.png", f"Classification Report: {name}")

 

#plot

plt.figure(figsize=(10, 8))


bars = plt.bar(cv_results.keys(), cv_results.values(), color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylabel('Mean CV Accuracy', fontsize=14)
plt.title('10-Fold Cross-Validation Performance', fontsize=16, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for i, v in enumerate(cv_results.values()):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=13, fontweight='bold')

plt.tight_layout() # Ensures nothing is cut off
plt.savefig(f"{RESULTS_DIR}/cv_accuracy_comparison.png")

