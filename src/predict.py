from utils import readImages
from featureExtraction import getFeatures
from timeit import default_timer as timer
import joblib

DATA_SOURCE = '../data'


def generateReport():
    # Load the saved final model
    model = joblib.load("../models/svm_model.pkl")

    # open the two files in write mode to override any old text or create them if they are not found
    resultFile = open("result.txt", "w")
    resultFile.close()
    timeFile = open("time.txt", "w")
    timeFile.close()
    # Now open the two files in append mode to append the needed results of each image
    resultFile = open("result.txt", "a")
    timeFile = open("time.txt", "a")
    imgs = readImages(DATA_SOURCE)
    for img in imgs:
        start = timer()
        features = getFeatures([img])
        prediction = model.predict(features)
        end = timer()
        roundedTime = round((end - start), 3)
        timeFile.write(str(roundedTime))
        timeFile.write("\n")
        resultFile.write(str(prediction[0]))
        resultFile.write("\n")
    timeFile.close()
    resultFile.close()
    print("Done! 🎉")


generateReport()
