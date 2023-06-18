import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from modelTraining import *
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression


def performanceMetrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def crossValidation(X, y):
    # Create a logistic regression model
    model = LogisticRegression()

    # Define the number of cross-validation folds
    k = 5

    # Create a cross-validation object
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    # Perform cross-validation and compute performance scores
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Print the performance scores
    print("Cross-Validation Performance Scores: ", scores)
    print("Mean Performance Score: ", scores.mean())

    # Plot a line plot of the performance scores
    plt.plot(range(1, k + 1), scores, marker="o")
    plt.xlabel("Cross-Validation Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Performance")
    plt.show()


def runPerformance(x_train, x_test, y_train, y_test):
    _, y_pred, y_true = SVM(x_train, x_test, y_train, y_test)
    performanceMetrics(y_true, y_pred)
