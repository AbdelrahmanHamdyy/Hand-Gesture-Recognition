import numpy as np
from utils import *
from featureExtraction import *
from modelTraining import *
from performance import runPerformance
from sklearn.metrics import accuracy_score
from predict import generateReport
import joblib


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
    true = []
    with open('../results.txt', 'r') as file:
        for line in file:
            true.append(line.strip())
    # -------------------------------------------
    # for i in range(6):
    #     if i == 2:
    #         true = true + ([str(i)] * 15)
    #     else:
    #         true = true + ([str(i)] * 17)
    accuracy = accuracy_score(true, predictions)
    print("Accuracy: {:.3f}%".format(accuracy * 100))


def sabry():
    # x, y = saveFeatures()
    # model = train(x, y)
    model = joblib.load("../models/svm_model.pkl")

    true = []
    with open('../results.txt', 'r') as file:
        for line in file:
            true.append(line.strip())
    imgs = readImages("../data4")
    x_test = getFeatures(imgs)
    test(x_test, true, model)


def run():
    # Extract Features
    x, y, features = saveFeatures()

    # Save features to CSV file
    saveToCSV(features, "../output.csv")

    # Split Training and Test Data
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)
        calcAccuracy(x_train, x_test, y_train, y_test)
    # runPerformance(x_train, x_test, y_train, y_test)
    # sample(x, y)
    # sabry()
    # Train and Test Model using different classifiers


if __name__ == "__main__":
    # run()
    sample()
