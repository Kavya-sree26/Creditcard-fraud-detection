# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# Step 1: Load the dataset
# Replace 'your_dataset.csv' with the actual dataset file path
data = pd.read_csv('creditcard.csv')
# Step 2: Data Preprocessing
# Scale the 'Amount' column
scaler = StandardScaler()
data['Scaled_Amount'] = scaler.fit_transform(data[['Amount']])
# Drop unnecessary columns
data = data.drop(['Time', 'Amount'], axis=1)  # Dropping 'Time' and original 'Amount'
# Define features and target
X = data.drop('Class', axis=1)  # Features: V1 to V28 and Scaled_Amount
y = data['Class']  # Target variable: Class (0: Non-Fraud, 1: Fraud)
# Step 3: Split data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Step 4: Feature Selection using RandomForest
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

RandomForestClassifier
?i
RandomForestClassifier(n_estimators=10, random_state=42)
# Select features with importance greater than a threshold
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
C:\Users\chakr\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\validation.py:2732: UserWarning: X has feature names, but SelectFromModel was fitted without feature names
  warnings.warn(
C:\Users\chakr\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\validation.py:2732: UserWarning: X has feature names, but SelectFromModel was fitted without feature names
  warnings.warn(
Step 5: Define and Train Models
# Model 1: Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_selected, y_train)
y_pred_lr = lr.predict(X_test_selected)
# Model 2: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train_selected, y_train)
y_pred_rf = rf_model.predict(X_test_selected)
# Model 3: Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train_selected, y_train)
y_pred_gb = gb.predict(X_test_selected)
# Model 4: Support Vector Classifier
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train_selected, y_train)
y_pred_svc = svc.predict(X_test_selected)
# Step 6: Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_true, y_pred)}")
# Step 7: Evaluate All Models
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_svc, "Support Vector Classifier")
Model: Logistic Regression
Accuracy: 0.9991
Precision: 0.8243
Recall: 0.6224
F1 Score: 0.7093
ROC AUC: 0.8111
Confusion Matrix:
 [[56851    13]
 [   37    61]]

Model: Random Forest
Accuracy: 0.9996
Precision: 0.9294
Recall: 0.8061
F1 Score: 0.8634
ROC AUC: 0.9030
Confusion Matrix:
 [[56858     6]
 [   19    79]]

Model: Gradient Boosting
Accuracy: 0.9984
Precision: 0.6087
Recall: 0.1429
F1 Score: 0.2314
ROC AUC: 0.5713
Confusion Matrix:
 [[56855     9]
 [   84    14]]

Model: Support Vector Classifier
Accuracy: 0.9994
Precision: 0.9571
Recall: 0.6837
F1 Score: 0.7976
ROC AUC: 0.8418
Confusion Matrix:
 [[56861     3]
 [   31    67]]
performance of ML MODELS
import matplotlib.pyplot as plt

# Models and Metric Values
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVC']

accuracy = [0.9991, 0.9996, 0.9984, 0.9994]
precision = [0.8243, 0.9294, 0.6087, 0.9571]
recall = [0.6224, 0.8061, 0.1429, 0.6837]
f1_score = [0.7093, 0.8634, 0.2314, 0.7976]
roc_auc = [0.8111, 0.9030, 0.5713, 0.8418]

# Function to Plot Individual Graphs with Custom Y-Axis Limits
def plot_metric_with_cutoff(metric_name, values, color, y_min, y_max):
    plt.figure(figsize=(8, 6))
    plt.bar(models, values, color=color, edgecolor='black', linewidth=1.2)
    plt.title(f'{metric_name} Comparison', fontsize=16, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12)
    plt.ylim(y_min, y_max)  # Set custom y-axis limits for clarity
    plt.xticks(fontsize=10, rotation=15)
    plt.yticks(fontsize=10)
    
    # Add value labels on each bar
    for i, val in enumerate(values):
        plt.text(i, val + (y_max - y_min) * 0.01, f"{val:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Improve Aesthetics
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot Individual Graphs for Each Metric with Different Colors and Limits
plot_metric_with_cutoff('Accuracy', accuracy, '#FF6F61', y_min=0.9975, y_max=1.0001)
plot_metric_with_cutoff('Precision', precision, '#6B5B95', y_min=0.5, y_max=1.0)
plot_metric_with_cutoff('Recall', recall, '#88B04B', y_min=0.1, y_max=1.0)
plot_metric_with_cutoff('F1 Score', f1_score, '#F7CAC9', y_min=0.2, y_max=1.0)
plot_metric_with_cutoff('ROC AUC', roc_auc, '#92A8D1', y_min=0.5, y_max=1.0)
