# evaluate.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

def evaluate_models(true_labels, all_preds, model_names):
    print("\nModel Performance Metrics:")
    for preds, name in zip(all_preds, model_names):
        print(f"\n{name} Evaluation:")
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
        rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"F1 Score : {f1:.2f}")
