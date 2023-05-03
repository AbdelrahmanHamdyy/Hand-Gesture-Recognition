import cv2 as cv
import numpy as np
from os import listdir, scandir
from os.path import isfile, join
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
# from numba import jit, cuda
from sklearn import metrics

INPUT_PATH = "../input/"


def extract_features(data_root_dir):
    features_list = []
    labels_list = []
    for outer_folder in listdir(data_root_dir):
        # outer folder is considered as the label
        for inner_file in scandir(join(root_dir, outer_folder)):
            image = cv.imread(inner_file.path, 0)
            print(inner_file.path)
            image_features = hog(resize(image, (128*4, 64*4)), orientations=9, pixels_per_cell=(14, 14),
                                 cells_per_block=(1, 1), visualize=False)
            features_list.append(image_features)
            labels_list.append(outer_folder)
    return features_list, labels_list


def train_model(features_list, labels_list):
    X_train, X_test, y_train, y_test = train_test_split(
        features_list, labels_list, test_size=0.33)
    # train using K-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    # get the model accuracy
    model_score = knn.score(X_test, y_test)
    print(model_score)
    # save trained model
    joblib.dump(knn, './models/knn_model_digits.pkl')


def train_model_svm(features_list, labels_list):
    svm_model = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(
        features_list, labels_list, test_size=0.33)
    # fitting x samples and y classes
    print("y train", y_train)
    svm_model.fit(X_train, y_train)
    # save trained model
    joblib.dump(svm_model, './models/svm_model_digits.pkl')
    y_pred = svm_model.predict(X_test)
    print(labels_list)
    print("==============================================")
    print(y_pred)
    print(y_test)
    print("Accuracy:", metrics.accuracy_score(y_pred, y_test))


if __name__ == '__main__':
    root_dir = "input"
    features_list, labels_list = extract_features(root_dir)
    train_model(features_list, labels_list)
