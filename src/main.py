import numpy as np
from utils import *
from featureExtraction import *
from modelTraining import *
from performance import runPerformance
from sklearn.metrics import accuracy_score
from predict import generateReport


def calcAccuracy(x_train, x_test, y_train, y_test):
    accSVM, _, _ = SVM(x_train, x_test, y_train, y_test)
    print("SVM Accuracy: {:.3f}%".format(accSVM * 100))


def sample():
    predictions = generateReport()
    true = []
    with open('../results.txt', 'r') as file:
        for line in file:
            true.append(line.strip())

    accuracy = accuracy_score(true, predictions)
    print("Accuracy: {:.3f}%".format(accuracy * 100))


def run():
    # Extract Features
    x, y, features = saveFeatures()

    # Save features to CSV file
    saveToCSV(features, "../output.csv")

    # Split Training and Test Data
    for i in range(15):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)
        calcAccuracy(x_train, x_test, y_train, y_test)

    # Performance
    runPerformance(x_train, x_test, y_train, y_test)

    # Training
    train(x, y)


if __name__ == "__main__":
    # run()
    generateReport()
    # sample()
