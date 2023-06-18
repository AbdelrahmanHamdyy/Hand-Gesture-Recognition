import cv2 as cv
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from preprocessing import *
import joblib


def readImages(dataPath, sort=False):
    files = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    x = []
    if sort:
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
    for fileName in files:
        img = cv.imread(join(dataPath, fileName))
        if (img is None):
            continue
        x.append(img)

    return x


def showImages(imgs, labels):
    if len(imgs) != len(labels):
        print("Length of images isn't the same as labels")
        return
    for i in range(len(imgs)):
        cv.imshow(labels[i], imgs[i])
        cv.waitKey(0)
    cv.destroyAllWindows()


def saveToCSV(features_dict, file):
    result = pd.DataFrame()

    for label, features_list in features_dict.items():
        df = pd.DataFrame(features_list)
        df['Class'] = label
        result = pd.concat([result, df])

    result.to_csv(file, mode='w+',  index=False)


def getAccuracySVM(file):
    # Load data from Excel file into a pandas DataFrame
    df = pd.read_csv(file)

    # Split the data into training and testing sets
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train an SVM model using the training data
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, '../models/svm_model.pkl')
    # Predict the labels for the test data
    accuracy = svm_model.score(X_test, y_test)
    print('Accuracy: {:.2f}'.format(accuracy))


def getAccuracyKNN(file):
    # Load data from Excel file into a pandas DataFrame
    df = pd.read_excel(file, sheet_name="Sheet1")

    # Split the data into training and testing sets
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train a KNN classifier on the training data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, '../models/knn_model.pkl')
    # Make predictions on the test data and evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))
