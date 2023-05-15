import numpy as np
from utils import *
from featureExtraction import *
from modelTraining import *
from performance import runPerformance
from sklearn.metrics import accuracy_score
from predict import generateReport


def calcAccuracy(x_train, x_test, y_train, y_test):
    # Train and Test Model using different classifiers
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


def sample(x=None, y=None):
    # train(x, y)
    predictions = generateReport()
    s0 = str(0)
    s1 = str(1)
    s2 = str(2)
    s3 = str(3)
    s4 = str(4)
    s5 = str(5)
    true = [s0, s0, s0, s0, s1, s1, s1, s2, s2, s2, s2,
            s3, s3, s3, s3, s3, s3, s4, s4, s4, s4, s5, s5, s5]
    accuracy = accuracy_score(true, predictions)
    print("Accuracy: {:.3f}%".format(accuracy * 100))


def run():
    # Extract Features
    # x, y = saveFeatures()

    # Save features to CSV file
    # saveToCSV(features, "../output.csv")

    # Split Training and Test Data
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # runPerformance(x_train, x_test, y_train, y_test)
    sample()

    # Train and Test Model using different classifiers
    # calcAccuracy(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    run()
