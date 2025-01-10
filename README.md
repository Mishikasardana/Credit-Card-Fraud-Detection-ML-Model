# Credit Card Fraud Detection

## Overview
This project aims to build a machine learning model to detect fraudulent credit card transactions. Fraud detection is critical for minimizing financial losses and ensuring secure transactions. The goal is to classify each transaction as fraudulent or legitimate using transaction details such as amounts, user information, and merchant details.

The project uses Python and libraries like **scikit-learn**, **pandas**, and **imbalanced-learn** for data preprocessing, modeling, and evaluation. Techniques such as **SMOTE (Synthetic Minority Oversampling Technique)** or undersampling are employed to handle the inherent class imbalance in fraud detection datasets.

---

## Table of Contents
1. [Features](#features)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Contributing](#contributing)

---

## Features
- Preprocessing steps to handle missing values, scaling, and balancing the dataset.
- Experimentation with various classification models:
  - Logistic Regression
  - Decision Trees
  - Random Forests
- Evaluation using key performance metrics such as:
  - Precision
  - Recall
  - F1-score
- Clear and concise documentation of methodology and results.

---

## Dataset
The dataset includes anonymized transaction information, typically containing:
- **Transaction Amount**
- **User Information**
- **Merchant Details**
- **Target Column** (0 for legitimate, 1 for fraudulent)

You can use datasets like [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

---

## Methodology
1. **Data Preprocessing:**
   - Handle missing values (if any).
   - Normalize/scale features for consistent input to the model.
   - Address class imbalance using techniques such as **SMOTE** or undersampling.

2. **Model Training:**
   - Experiment with models like Logistic Regression, Decision Trees, and Random Forests.
   - Split data into training and testing sets for unbiased evaluation.

3. **Evaluation Metrics:**
   - Use metrics like **precision**, **recall**, **F1-score**, and **AUC-ROC** to measure model performance.

4. **Hyperparameter Tuning:**
   - Use grid search or random search to optimize model parameters for better performance.

---

## Requirements
Install the following libraries:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Place the dataset in the `data/` directory.
4. Run the preprocessing and training script:
   ```bash
   python train_model.py
   ```
5. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```
6. Predict on new transactions:
   ```bash
   python predict.py --input new_transactions.csv
   ```

---

## Evaluation
The model will be evaluated using the following metrics:
- **Precision:** Measures the proportion of true fraud predictions out of all fraud predictions.
- **Recall:** Measures the proportion of actual fraud transactions correctly identified.
- **F1-Score:** Harmonic mean of precision and recall, ensuring a balance between false positives and false negatives.
- **AUC-ROC Curve:** Measures the model's ability to distinguish between classes.

Results will be saved in the `results/` directory.

---

## Results
The model performance for the selected dataset is as follows:
Logistic Regression Classifier:
Accuracy: 0.9506726457399103
Precision: 0.9702970297029703
Recall: 0.6533333333333333
F1 Score: 0.7808764940239044

Decision Tree Classifier:
Accuracy: 0.9204510447901907
Precision: 0.8125
Recall: 0.6333333333333333
F1 Score: 0.7114093959731543

Random Forest Classifier:
Accuracy: 0.9757847533632287
Precision: 0.984251968503937
Recall: 0.8333333333333334
F1 Score: 0.9025270758122743
---

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork.
4. Submit a pull request for review.

---
