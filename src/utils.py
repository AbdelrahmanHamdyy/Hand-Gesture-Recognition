import cv2 as cv
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from preprocessing import *


def readImages(dataPath, num=0):
    files = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    counter = 0
    x = []
    for fileName in files:
        img = cv.imread(dataPath + "/" + fileName)
        if (img is None):
            continue
        # img = preprocess(img)
        # cv.imwrite("../Dataset-output" + "/"+ str(num)+"/"+ str(fileName),img)
        # print("===============",fileName+"=======================")
        # print(img)
        x.append(img)
        # counter=counter+1
        # if(counter>2):
        #     break
    return x


def showImages(imgs, labels):
    if len(imgs) != len(labels):
        print("Length of images isn't the same as labels")
        return
    for i in range(len(imgs)):
        cv.imshow(labels[i], imgs[i])
        cv.waitKey(0)
    cv.destroyAllWindows()


def saveToExcel(features_dict, file):
    result = pd.DataFrame()
    with pd.ExcelWriter(file, mode="a", engine="openpyxl", if_sheet_exists='replace') as writer:
        for label, features_list in features_dict.items():
            df = pd.DataFrame(features_list)
            df['Class'] = label
            result = pd.concat([result, df])
        result.to_excel(writer, sheet_name="Sheet1", index=False)


def saveToCSV(features_dict, file):
    result = pd.DataFrame()

    for label, features_list in features_dict.items():
        df = pd.DataFrame(features_list)
        df['Class'] = label
        result = pd.concat([result, df])

    result.to_csv(file, mode='a', header=False)


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

    # Make predictions on the test data and evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))


def getAccuracySVM(file):
    # Load data from Excel file into a pandas DataFrame
    df = pd.read_excel(file, sheet_name="Sheet1")

    # Split the data into training and testing sets
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train an SVM model using the training data
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)

    # Predict the labels for the test data
    # y_pred = svm_model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    accuracy = svm_model.score(X_test, y_test)
    print('Accuracy: {:.2f}'.format(accuracy))
