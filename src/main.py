import numpy as np
from utils import *
from featureExtraction import *
from modelTraining import *
from performance import runPerformance
from sklearn.metrics import accuracy_score
from predict import generateReport


def calcAccuracy(x_train, x_test, y_train, y_test):
    # Train and Test Model using different classifiers
    print("Testing...")
    accSVM, _, _ = SVM(x_train, x_test, y_train, y_test)
    # accKNN3 = KNN(x_train, x_test, y_train, y_test, 3)
    # accKNN5 = KNN(x_train, x_test, y_train, y_test, 5)
    # accGBC = GBC(x_train, x_test, y_train, y_test)
    # accSVR = trainSVR(x_train, x_test, y_train, y_test)
    # accRF = randomForest(x_train, x_test, y_train, y_test)
    # accBayes = bayes(x_train, x_test, y_train, y_test)
    # accLR = logisticRegression(x_train, x_test, y_train, y_test)
    # accDT = decisionTree(x_train, x_test, y_train, y_test)

    # Print accuracies
    print("SVM Accuracy: {:.3f}%".format(accSVM * 100))
    # print('KNN3 Accuracy: {:.3f}'.format(accKNN3))
    # print('KNN5 Accuracy: {:.3f}'.format(accKNN5))
    # print('GBC Accuracy: {:.3f}'.format(accGBC))
    # print('SVR Accuracy: {:.3f}'.format(accSVR))
    # print('RF Accuracy: {:.3f}'.format(accRF))
    # print('Bayes Accuracy: {:.3f}'.format(accBayes))
    # print("LR Accuracy: {:.3f}".format(accLR))
    # print('DT Accuracy: {:.3f}'.format(accDT))

    # Print best accuracy
    # print("------------------------------")
    # accuracy = max(accSVM, accLR)
    # print("Best Accuracy: {:.3f}".format(accSVM))


def sample(x, y):
    train(x, y)
    predictions = generateReport()
    true = []
    for i in range(6):
        if i == 2:
            true = true + ([str(i)] * 15)
        else:
            true = true + ([str(i)] * 17)
    accuracy = accuracy_score(true, predictions)
    print("Accuracy: {:.3f}%".format(accuracy * 100))


def run():
    # Extract Features
    x, y = saveFeatures()

    # Save features to CSV file
    # saveToCSV(features, "../output.csv")

    # Split Training and Test Data
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # runPerformance(x_train, x_test, y_train, y_test)
    sample(x, y)

    # Train and Test Model using different classifiers
    # calcAccuracy(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    run()
