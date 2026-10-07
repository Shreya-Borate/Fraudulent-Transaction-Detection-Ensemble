# =============================================================================
# FRAUDULENT TRANSACTION DETECTION USING ENSEMBLE LEARNING
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =============================================================================
# STEP 1 : LOAD THE DATASET
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 1 : LOAD THE DATASET")
print("=" * 80)



df = pd.read_csv("../data/Fraudulent_Transaction_Detection.csv")

print("\nDataset loaded successfully.")

print("\nFirst 5 records :")
print(df.head())


# =============================================================================
# STEP 2 : UNDERSTAND THE DATASET
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 2 : UNDERSTAND THE DATASET")
print("=" * 80)

print("\nDataset Shape :")
print(df.shape)

print("\nColumn Names :")
print(list(df.columns))

print("\nDataset Information :")
df.info()

print("\nStatistical Information :")
print(df.describe())


# =============================================================================
# STEP 3 : CHECK FOR MISSING VALUES
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 3 : CHECK FOR MISSING VALUES")
print("=" * 80)

print("\nMissing values in each column :")
print(df.isnull().sum())


# =============================================================================
# STEP 4 : CHECK TARGET DISTRIBUTION
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 4 : CHECK TARGET DISTRIBUTION")
print("=" * 80)

print("\nFraud Class Distribution :")
print(df["Fraud"].value_counts())

print("\nFraud Class Percentage :")
print(df["Fraud"].value_counts(normalize=True) * 100)


# =============================================================================
# STEP 5 : VISUALIZE TARGET DISTRIBUTION
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 5 : VISUALIZE TARGET DISTRIBUTION")
print("=" * 80)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Fraud"
)

plt.title("Fraudulent Transaction Class Distribution")
plt.xlabel("Fraud")
plt.ylabel("Number of Transactions")

plt.xticks(
    [0, 1],
    ["Normal Transaction", "Fraudulent Transaction"]
)

plt.tight_layout()
plt.show()


# =============================================================================
# STEP 6 : SEPARATE INPUT AND OUTPUT VARIABLES
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 6 : SEPARATE INPUT AND OUTPUT VARIABLES")
print("=" * 80)

# X : Independent Variables / Features
# Y : Dependent Variable / Target

X = df.drop("Fraud", axis=1)
Y = df["Fraud"]

print("\nFeature Columns :")
print(list(X.columns))

print("\nTarget Variable :")
print("Fraud")

print("\nFeature Shape :", X.shape)
print("Target Shape :", Y.shape)


# =============================================================================
# STEP 7 : TRAIN-TEST SPLIT
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 7 : TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("\nTraining data shape :")
print(X_train.shape)

print("\nTesting data shape :")
print(X_test.shape)

print("\nTraining target shape :")
print(Y_train.shape)

print("\nTesting target shape :")
print(Y_test.shape)


# =============================================================================
# STEP 8 : BUILD DECISION TREE CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 8 : BUILD DECISION TREE CLASSIFIER")
print("=" * 80)

DecisionTreeModel = DecisionTreeClassifier(
    random_state=42
)

print("\nDecision Tree Classifier created successfully.")


# =============================================================================
# STEP 9 : BUILD BAGGING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 9 : BUILD BAGGING CLASSIFIER")
print("=" * 80)

BaggingModel = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)

print("\nBagging Classifier created successfully.")


# =============================================================================
# STEP 10 : BUILD RANDOM FOREST CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 10 : BUILD RANDOM FOREST CLASSIFIER")
print("=" * 80)

RandomForestModel = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nRandom Forest Classifier created successfully.")


# =============================================================================
# STEP 11 : BUILD ADABOOST CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 11 : BUILD ADABOOST CLASSIFIER")
print("=" * 80)

AdaBoostModel = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

print("\nAdaBoost Classifier created successfully.")


# =============================================================================
# STEP 12 : TRAIN INDIVIDUAL MODELS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 12 : TRAIN INDIVIDUAL MODELS")
print("=" * 80)

DecisionTreeModel.fit(X_train, Y_train)

BaggingModel.fit(X_train, Y_train)

RandomForestModel.fit(X_train, Y_train)

AdaBoostModel.fit(X_train, Y_train)

print("\nAll individual models trained successfully.")


# =============================================================================
# STEP 13 : CREATE VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 13 : CREATE VOTING CLASSIFIER")
print("=" * 80)

VotingModel = VotingClassifier(
    estimators=[
        ("Decision Tree", DecisionTreeModel),
        ("Bagging", BaggingModel),
        ("Random Forest", RandomForestModel),
        ("AdaBoost", AdaBoostModel)
    ],
    voting="soft"
)

print("\nVoting Classifier created successfully.")


# =============================================================================
# STEP 14 : TRAIN VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 14 : TRAIN VOTING CLASSIFIER")
print("=" * 80)

VotingModel.fit(X_train, Y_train)

print("\nVoting Classifier trained successfully.")


# =============================================================================
# STEP 15 : MAKE PREDICTIONS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 15 : MAKE PREDICTIONS")
print("=" * 80)

DecisionTreePred = DecisionTreeModel.predict(X_test)

BaggingPred = BaggingModel.predict(X_test)

RandomForestPred = RandomForestModel.predict(X_test)

AdaBoostPred = AdaBoostModel.predict(X_test)

VotingPred = VotingModel.predict(X_test)

print("\nPredictions generated successfully.")


# =============================================================================
# STEP 16 : EVALUATION FUNCTION
# =============================================================================

def EvaluateModel(ModelName, YActual, YPred):

    Accuracy = accuracy_score(YActual, YPred)

    Precision = precision_score(
        YActual,
        YPred,
        zero_division=0
    )

    Recall = recall_score(
        YActual,
        YPred,
        zero_division=0
    )

    F1 = f1_score(
        YActual,
        YPred,
        zero_division=0
    )

    print("\n")
    print("-" * 80)
    print(ModelName)
    print("-" * 80)

    print("Accuracy  :", Accuracy)
    print("Precision :", Precision)
    print("Recall    :", Recall)
    print("F1 Score  :", F1)

    print("\nClassification Report :")
    print(
        classification_report(
            YActual,
            YPred,
            target_names=[
                "Normal Transaction",
                "Fraudulent Transaction"
            ],
            zero_division=0
        )
    )

    return {
        "Model": ModelName,
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Recall": Recall,
        "F1 Score": F1
    }


# =============================================================================
# STEP 17 : EVALUATE ALL MODELS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 17 : EVALUATE ALL MODELS")
print("=" * 80)

DecisionTreeResult = EvaluateModel(
    "Decision Tree",
    Y_test,
    DecisionTreePred
)

BaggingResult = EvaluateModel(
    "Bagging",
    Y_test,
    BaggingPred
)

RandomForestResult = EvaluateModel(
    "Random Forest",
    Y_test,
    RandomForestPred
)

AdaBoostResult = EvaluateModel(
    "AdaBoost",
    Y_test,
    AdaBoostPred
)

VotingResult = EvaluateModel(
    "Voting Classifier",
    Y_test,
    VotingPred
)


# =============================================================================
# STEP 18 : FINAL MODEL COMPARISON
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 18 : FINAL MODEL COMPARISON")
print("=" * 80)

ModelResults = pd.DataFrame([
    DecisionTreeResult,
    BaggingResult,
    RandomForestResult,
    AdaBoostResult,
    VotingResult
])

print("\nFinal Model Comparison :")
print(ModelResults)

# Round the evaluation metrics
ComparisonTable = ModelResults.copy()

Metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

ComparisonTable[Metrics] = ComparisonTable[Metrics].round(4)

print("\nFinal Model Comparison - Rounded :")
print(ComparisonTable)


# =============================================================================
# STEP 19 : MODEL COMPARISON VISUALIZATION
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 19 : MODEL COMPARISON VISUALIZATION")
print("=" * 80)

ComparisonPlot = ModelResults.set_index("Model")[Metrics]

ComparisonPlot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Fraud Detection Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")

plt.ylim(0, 1.1)

plt.xticks(rotation=20)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.show()


# =============================================================================
# STEP 20 : CONFUSION MATRIX
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 20 : CONFUSION MATRIX")
print("=" * 80)

ConfusionMatrix = confusion_matrix(
    Y_test,
    VotingPred
)

print("\nVoting Classifier Confusion Matrix :")
print(ConfusionMatrix)

plt.figure(figsize=(7, 5))

sns.heatmap(
    ConfusionMatrix,
    annot=True,
    fmt="d",
    xticklabels=[
        "Normal Transaction",
        "Fraudulent Transaction"
    ],
    yticklabels=[
        "Normal Transaction",
        "Fraudulent Transaction"
    ]
)

plt.title("Voting Classifier - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# =============================================================================
# STEP 21 : BEST MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 21 : BEST MODEL")
print("=" * 80)

BestModel = ModelResults.loc[
    ModelResults["F1 Score"].idxmax()
]

print("\nBest Model based on F1 Score :")
print(BestModel)


# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n")
print("=" * 80)
print("FRAUDULENT TRANSACTION DETECTION PROJECT COMPLETED")
print("=" * 80)