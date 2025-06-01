# 📊 ML Model Performance Evaluation (From Scratch)

This story walks through the lifecycle of evaluating a **machine learning model**, especially in the context of **binary classification**, and connects key performance concepts with hands-on implementations in both **Python** and **C++**.

---

## 📖 The Story of Building and Evaluating a Model

### 🏁 It starts with a decision point…
Imagine you're building a model to classify whether an email is **spam** or **not spam**. Once the model is trained, how do you know it's any good?

You begin by measuring its **accuracy**, which seems intuitive at first: the number of correct predictions over total predictions.

### ❗ The accuracy trap in unbalanced classes
What if only 5% of emails are spam? A model that always predicts "not spam" would be **95% accurate** — and yet, completely useless.

This brings us to the need for **better metrics**.

### 🧪 Model Training and Validation
You split your dataset into **training**, **validation**, and **test** sets:
- **Training set** to fit the model.
- **Validation set** to tune **hyperparameters**.
- **Test set** to evaluate generalization.

You consider **k-Fold Cross Validation** to train and validate on different subsets for more robustness. You even explore **Leave-One-Out Cross Validation** (LOOCV), especially when data is scarce.

### 🧮 Confusion Matrix: Your evaluation dashboard
The confusion matrix gives detailed insight:
```
                 Actual
              |  1   |  0  |
Predicted |-----------|
        1    | TP  | FP |
        0    | FN  | TN |
```

From here, we get:
- **Sensitivity (Recall)** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)
- **Precision** = TP / (TP + FP)
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)

These tell us **how well the model detects positives**, avoids false alarms, and balances the two.

### 📈 ROC & AUC
By plotting the **True Positive Rate vs. False Positive Rate**, the **ROC Curve** shows performance across thresholds. The **AUC** (Area Under Curve) summarizes this into a single score — closer to 1 is better.

### 🔧 Hyperparameter Tuning
You optimize things like:
- Number of trees in a random forest
- Depth of a decision tree
- Learning rate in gradient boosting

You use the **validation set or k-fold** to select these, then test on unseen data.

### 🌍 Generalization
Ultimately, you want a model that doesn't just memorize — it must generalize to new, real-world data. That’s the end goal.

---

## 📉 Dynamic Evaluation: Varying Train/Validation/Test Splits

To better understand how **training set size** affects model performance, we simulate multiple scenarios with:
- A fixed dataset of 100 samples
- Varying training sizes (e.g., 50 to 70)
- Automatically adjusting validation and test sizes

For each configuration, we compute:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

And we compare results across:
- **Training Set**
- **Validation Set**
- **Test Set**

### 🔍 Key Insights from Visualization
- As **training set size increases**, model performance becomes more stable
- **Validation and test sets**, being smaller, show more variability in metrics
- **F1 Score** and **Recall** tend to fluctuate the most, especially under class imbalance

This visualization helps in understanding how the split of your data impacts model evaluation and can guide decisions on how much data to allocate for training vs. validation.
