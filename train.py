import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------

data = pd.read_csv("data/fake_real_datasett.csv", engine="python", on_bad_lines="skip")

print("Dataset Loaded Successfully")
print(f"Total samples: {len(data)}")

# ---------------------------
# 2️⃣ Clean Dataset
# ---------------------------

# Remove empty rows
data = data.dropna()

# Convert label to numeric
data['label'] = pd.to_numeric(data['label'], errors='coerce')

# Remove invalid labels
data = data.dropna(subset=['label'])

# Convert label to int
data['label'] = data['label'].astype(int)

# Keep only 0 and 1 labels
data = data[data['label'].isin([0,1])]
data = data.groupby('label').apply(lambda x: x.sample(data['label'].value_counts().min())).reset_index(drop=True)
print("\nCleaned Label Distribution:")
print(data['label'].value_counts())

# ---------------------------
# 3️⃣ Define Features
# ---------------------------

X = data['text']
y = data['label']

# ---------------------------
# 4️⃣ Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------
# 5️⃣ TF-IDF Vectorization
# ---------------------------

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTF-IDF Features: {X_train_tfidf.shape[1]}")

# ---------------------------
# 6️⃣ Define Models
# ---------------------------

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=15),
    "Passive Aggressive": PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000)
}

best_accuracy = 0
best_model = None
best_model_name = ""

# ---------------------------
# 7️⃣ Create Output Folders
# ---------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/confusion_matrices", exist_ok=True)
os.makedirs("outputs/roc_curves", exist_ok=True)
os.makedirs("trainoutput", exist_ok=True)

results = []

# print("\n" + "="*70)
# print("🚀 TRAINING MODELS - COMPARING TRAIN vs TEST ACCURACY")
# print("="*70)

# ---------------------------
# 8️⃣ Train Models
# ---------------------------

for name, model in models.items():

    # print(f"\n{'='*70}")
    print(f" Model: {name}")
    # print(f"{'='*70}")

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # ✅✅✅ IMPORTANT: Calculate BOTH Training and Testing Accuracy ✅✅✅
    
    # 1️⃣ TRAINING ACCURACY (on training data)
    y_train_pred = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # 2️⃣ TESTING ACCURACY (on test data)
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Print both accuracies
    print(f"\n Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f" Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Calculate t he difference
    accuracy_diff = train_accuracy - test_accuracy
    print(f"\n  Difference: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
    
    # Interpretation
    if accuracy_diff > 0.1:
        print("  Warning: Model might be OVERFITTING (train accuracy >> test accuracy)")
    elif accuracy_diff < -0.05:
        print("  Warning: Something unusual (test accuracy > train accuracy)")
    else:
        print(" Good: Model is generalizing well")

    # Classification Report (on test data)
    report = classification_report(y_test, y_test_pred)
    print(f"\n Classification Report (Test Data):")
    print(report)

    # ---------------------------
    # AUC Score
    # ---------------------------

    try:
        y_prob = model.predict_proba(X_test_tfidf)[:,1]
    except:
        y_prob = model.decision_function(X_test_tfidf)

    auc_score = roc_auc_score(y_test, y_prob)

    print(f" AUC Score: {auc_score:.4f}")
    
    # Save results
    results.append({
        "Model": name,
        "Training_Accuracy": train_accuracy,
        "Testing_Accuracy": test_accuracy,
        "Accuracy_Difference": accuracy_diff,
        "AUC_Score": auc_score,
        "Report": report
    })

    # Select best model based on test accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = model
        best_model_name = name

    # ---------------------------
    # Confusion Matrix
    # ---------------------------

    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)

    plt.title(f"{name} Confusion Matrix\n(0=True News, 1=Fake News)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrices/{name.replace(' ', '_')}_cm.png", dpi=300)

    plt.close()
    
    # ---------------------------
    # ROC Curve
    # ---------------------------

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,6))

    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")

    plt.plot([0,1], [0,1], linestyle="--", color='gray', linewidth=1)

    plt.xlabel("False Positive Rate", fontsize=12)

    plt.ylabel("True Positive Rate", fontsize=12)

    plt.title(f"{name} ROC Curve", fontsize=14, fontweight='bold')

    plt.legend(loc="lower right", fontsize=11)
    
    plt.grid(alpha=0.3)
    
    plt.tight_layout()

    plt.savefig(f"outputs/roc_curves/{name.replace(' ', '_')}_roc.png", dpi=300)

    plt.close()

print("\n" + "="*70)

# ---------------------------
# 9️⃣ Save Detailed Results
# ---------------------------

with open("trainoutput/model_results.txt", "w", encoding='utf-8') as f:

    # f.write("="*70 + "\n")
    # f.write("🎯 FAKE NEWS DETECTION - MODEL COMPARISON REPORT\n")
    # f.write("="*70 + "\n\n")

    for r in results:

        # f.write(f"\n{'='*70}\n")
        f.write(f" Model: {r['Model']}\n")
        # f.write(f"{'='*70}\n\n")
        
        f.write(f" Training Accuracy:  {r['Training_Accuracy']:.4f} ({r['Training_Accuracy']*100:.2f}%)\n")
        f.write(f" Testing Accuracy:   {r['Testing_Accuracy']:.4f} ({r['Testing_Accuracy']*100:.2f}%)\n")
        f.write(f"  Accuracy Difference: {r['Accuracy_Difference']:.4f} ({r['Accuracy_Difference']*100:.2f}%)\n")
        f.write(f" AUC Score:          {r['AUC_Score']:.4f}\n\n")

        f.write(" Classification Report (Test Data):\n")
        # f.write("-"*70 + "\n")
        f.write(r["Report"])

        f.write("\n" + "="*70 + "\n")

# print("\n✅ Model comparison report saved to: trainoutput/model_results.txt")

# ---------------------------
# 🔟 Save Best Model
# ---------------------------

pickle.dump(best_model, open("models/best_model.pkl", "wb"))

pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# print("\n" + "="*70)
print(" BEST MODEL SELECTED")
# print("="*70)
print(f"Model Name: {best_model_name}")
print(f"Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("\n Best model and vectorizer saved successfully")
# print("   - models/best_model.pkl")
# print("   - models/vectorizer.pkl")
# print("="*70)

# print("\n🎉 Training Complete! Check 'trainoutput/model_results.txt' for detailed comparison.")
