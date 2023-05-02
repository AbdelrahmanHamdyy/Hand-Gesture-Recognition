import cv2 as cv
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def readImages(dataPath):
    files = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    counter=0
    x = []
    for fileName in files:
        img = cv.imread(dataPath + "/" + fileName, 0)
        x.append(img)
        counter=counter+1
        if(counter>20):
            break
    return x

def showImages(imgs, labels):
    if len(imgs) != len(labels):
        print("Length of images isn't the same as labels")
        return
    for i in range(len(imgs)):
        cv.imshow(labels[i], imgs[i])
        cv.waitKey(0)
    cv.destroyAllWindows()

## save to excel file
def saveToExcel(features_dict,file):
    result= pd.DataFrame()
    with pd.ExcelWriter(file,mode="a",engine="openpyxl",if_sheet_exists='replace') as writer:
        for label, features_list in features_dict.items():
            df = pd.DataFrame(features_list)
            df['Class'] = label
            result=pd.concat([result,df])
        result.to_excel(writer,sheet_name="Sheet1",index=False)

## get accuracy by KNN
def getAccuracy(file):
    # Load data from Excel file into a pandas DataFrame
    df = pd.read_excel(file, sheet_name="Sheet1")

    # Split the data into training and testing sets
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a KNN classifier on the training data
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Make predictions on the test data and evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))