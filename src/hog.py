import cv2 as cv
import numpy as np
from utils import readImages
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from utils import *
from preprocess import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

INPUT_PATH = "../Dataset_0-5/"
featuresDict = {}


def lbp(img, radius=3, n_points=8):
    """
    Compute Local Binary Pattern (LBP) features for an image.
    Parameters:
        img: 2D numpy array representing the image
        radius: radius of the circular LBP sampling region (default: 3)
        n_points: number of sampling points in the circular region (default: 8)
    Returns:
        2D numpy array representing the LBP features
    """
    lbp_img = local_binary_pattern(img, n_points, radius, "uniform")
    hist, _ = np.histogram(
        lbp_img.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """
    Compute Histogram of Oriented Gradients (HOG) features for an image.
    Parameters:
        img: 2D numpy array representing the image
        orientations: number of gradient orientation bins (default: 9)
        pixels_per_cell: size of a cell in pixels (default: (8, 8))
        cells_per_block: number of cells in each block (default: (3, 3))
    """
    hog_feats = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        feature_vector=True,
    )
    return hog_feats


def applyPCA(features):
    pca = PCA(n_components=172)  # keep the top N principal components
    pca.fit(features)
    featuresPCA = pca.transform(features)
    return featuresPCA


def getFeatures(x_train):
    features = []
    for img in x_train:
        img = preprocess(img)

        # fd = hog_features(resize(img, (128*4, 64*4)))
        fd = hog(resize(img, (128*4, 64*4)), orientations=9,
                 pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        # fd = lbp(resize(img, (128*4, 64*4)))
        features.append(fd)

    # fd_PCA = applyPCA(features)
    return features


def saveFeatures():
    x = []
    y = []
    for i in range(6):
        print(i)
        print("------------------------------")
        INPUT_PATH = "../Dataset_0-5/"+str(i)
        imgs = readImages(INPUT_PATH,i)
        features = getFeatures(imgs)

        featuresDict[str(i)] = features
        x = x + features
        y = y + ([str(i)] * len(features))

    return x, y


def KNN(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    # Make predictions on the test data and evaluate the model
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))


def SVM(x_train, x_test, y_train, y_test):
    # Train an SVM model using the training data
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(x_train, y_train)

    # Predict labels for the test data
    accuracy = svm_model.score(x_test, y_test)
    print('Accuracy: {:.2f}'.format(accuracy))


def run():
    x, y = saveFeatures()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    SVM(x_train, x_test, y_train, y_test)
    # saveToCSV(featuresDict,"../output.csv")


if __name__ == '__main__':
    run()
