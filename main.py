import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Check for missing values and handle them if any
if df.isnull().values.any():
    df = df.dropna()

# Exploratory Data Analysis
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

# Preprocess data: Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler to normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define a function to train and evaluate models with cross-validation and hyperparameter tuning
def train_and_evaluate_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_smote, y_train_smote)
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC Score: {roc_auc}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    return best_model

# Train and evaluate Logistic Regression classifier
print("Logistic Regression:")
lr_param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
best_lr = train_and_evaluate_model(LogisticRegression(max_iter=1000), lr_param_grid)

# Train and evaluate Decision Tree classifier
print("\nDecision Tree:")
dt_param_grid = {'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10]}
best_dt = train_and_evaluate_model(DecisionTreeClassifier(), dt_param_grid)

# Train and evaluate Random Forest classifier
print("\nRandom Forest:")
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
best_rf = train_and_evaluate_model(RandomForestClassifier(), rf_param_grid)

# Train and evaluate Gradient Boosting classifier
print("\nGradient Boosting:")
gb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
best_gb = train_and_evaluate_model(GradientBoostingClassifier(), gb_param_grid)

# Save the best models
joblib.dump(best_lr, 'best_logistic_regression_model.pkl')
joblib.dump(best_dt, 'best_decision_tree_model.pkl')
joblib.dump(best_rf, 'best_random_forest_model.pkl')
joblib.dump(best_gb, 'best_gradient_boosting_model.pkl')
