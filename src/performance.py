import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from modelTraining import *


def checkPreformance(y_true, y_pred):
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


def runPerformance(x_train, x_test, y_train, y_test):
    y_pred, y_true = SVM(x_train, x_test, y_train, y_test)
    metrics = checkPreformance(y_true, y_pred)
    metrics.plot_confusion_matrix()


if __name__ == '__main__':
    runPerformance()
