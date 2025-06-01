import matplotlib.pyplot as plt
import pandas as pd
import random

# Generate random predictions and labels
def generate_binary_data(size=100, positive_ratio=0.3, model_accuracy=0.85):
    labels = [1 if i < int(size * positive_ratio) else 0 for i in range(size)]
    random.shuffle(labels)
    predictions = []
    for label in labels:
        if random.random() < model_accuracy:
            predictions.append(label)
        else:
            predictions.append(1 - label)
    return predictions, labels

# Evaluate performance metrics
def evaluate_model(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, specificity, f1

# Evaluate model with varying splits
def evaluate_with_varying_splits():
    dataset_size = 100
    results = []

    for train_size in range(50, 81, 5):
        val_test_size = dataset_size - train_size
        val_size = val_test_size // 2
        test_size = val_test_size - val_size

        preds, labels = generate_binary_data(dataset_size)
        train_preds = preds[:train_size]
        train_labels = labels[:train_size]
        val_preds = preds[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        test_preds = preds[train_size + val_size:]
        test_labels = labels[train_size + val_size:]

        for name, p, l in [("Train", train_preds, train_labels), ("Validation", val_preds, val_labels), ("Test", test_preds, test_labels)]:
            acc, prec, rec, spec, f1 = evaluate_model(p, l)
            results.append({
                "Set": name,
                "Train Size": train_size,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "Specificity": spec,
                "F1 Score": f1
            })

    return pd.DataFrame(results)

# Create and visualize metrics
df = evaluate_with_varying_splits()
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    for set_type in df["Set"].unique():
        subset = df[df["Set"] == set_type]
        ax.plot(subset["Train Size"], subset[metric], marker='o', label=set_type)
    ax.set_title(f"{metric} vs. Training Size")
    ax.set_xlabel("Training Size")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
