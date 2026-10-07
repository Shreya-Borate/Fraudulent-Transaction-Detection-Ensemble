# Fraudulent Transaction Detection using Ensemble Learning

## 📌 Project Overview

This project focuses on detecting potentially fraudulent financial transactions using Machine Learning classification and ensemble learning techniques.

The objective is to classify transactions into:

- Normal Transaction
- Fraudulent Transaction

The project investigates and compares multiple ensemble-based approaches and recommends the most suitable model based on several evaluation metrics.

The following models are implemented:

1. Decision Tree
2. Bagging Classifier
3. Random Forest Classifier
4. AdaBoost Classifier
5. Voting Classifier

---

## 🎯 Objective

The main objectives of this project are:

- To understand transaction-level financial data.
- To identify patterns associated with fraudulent transactions.
- To build multiple Machine Learning classifiers.
- To implement different ensemble learning approaches.
- To compare model performance.
- To evaluate models using Accuracy, Precision, Recall and F1 Score.
- To analyze the Confusion Matrix.
- To recommend the most suitable model for fraud detection.

---

## 📊 Dataset

The dataset contains transaction-level information related to potentially fraudulent financial activity.

### Input Features

| Feature | Description |
|---|---|
| TransactionAmount | Amount involved in the transaction |
| TransactionHour | Hour at which the transaction occurred |
| AccountAgeMonths | Age of the account in months |
| PreviousTransactions | Number of previous transactions |
| LocationDifferenceKm | Difference between transaction location and expected location |
| DeviceType | Encoded device type |
| FailedLoginAttempts | Number of failed login attempts |

### Target Variable

| Value | Meaning |
|---|---|
| 0 | Normal Transaction |
| 1 | Fraudulent Transaction |

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Dataset Understanding
   ↓
Missing Value Check
   ↓
Target Distribution Analysis
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Decision Tree
   ↓
Bagging
   ↓
Random Forest
   ↓
AdaBoost
   ↓
Voting Classifier
   ↓
Predictions
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Confusion Matrix
   ↓
Best Model Recommendation