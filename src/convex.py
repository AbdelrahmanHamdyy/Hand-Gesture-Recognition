import cv2 as cv
import numpy as np
import math


def imageFiltering(frame):
    # area of intereset(hand)
    roi = frame[y : y + h, x : x + w]

    # applying gaussian blurr to reduce the noise
    blur = cv.GaussianBlur(roi, (5, 5), 0)
    # converting from coloured to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # applying a mask which makes skin color white and others black
    mask = cv.inRange(hsv, np.array([2, 50, 50]), np.array([20, 255, 255]))

    kernel = np.ones((5, 5))
    # reducing noise
    filtered = cv.GaussianBlur(mask, (3, 3), 0)
    ret, thresh = cv.threshold(filtered, 127, 255, 0)  # thesholding the image
    thesh = cv.GaussianBlur(thresh, (5, 5), 0)  # reducing the noise
    # finding contours in the image. Will be used later in complex hull algorithm
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return roi, thresh, contours


cap = cv.VideoCapture(0)

# dimensions of the box
x = 100
y = 100
w = 200
h = 200

while True:
    ret, frame = cap.read()  # read video frame by frame
    # create a rectangle around roi
    cv.rectangle(frame, (y, x), (y + h, x + w), (0, 255, 0), 2)

    roi, thresh, contours = imageFiltering(frame)  # getting the filtered image

    # blank image which will be used to show the contours and defects
    drawing = np.zeros(roi.shape, np.uint8)
    try:
        # finding contour with max area
        contour = max(contours, key=lambda x: cv.contourArea(x), default=0)

        # convex hull. This creates a convex polygon.
        hull = cv.convexHull(contour)

        # draw contours
        cv.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # finding defects in the convex polygon formed using convex hull algorithm
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)

        count_defects = 0  # defaults initially set to 0

        # finding defects and displaying them on the image
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]  # defect returns 4 arguments
            # using start, end, far to find the defects location
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # finding the angle of the defect using cosine law
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 180) / 3.14

            # we know, angle between 2 fingers is within 90 degrees.
            # so anything greater than that isn;t considered
            if angle <= 90:
                count_defects += 1
                cv.circle(drawing, far, 5, [0, 0, 255], -1)  # displaying defect

            cv.line(drawing, start, end, [0, 255, 0], 2)

        if count_defects == 0:
            cv.putText(
                frame, "ONE", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2
            )
        elif count_defects == 1:
            cv.putText(
                frame, "TWO", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2
            )
        elif count_defects == 2:
            cv.putText(
                frame, "THREE", (5, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2
            )
        elif count_defects == 3:
            cv.putText(
                frame, "FOUR", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2
            )
        elif count_defects == 4:
            cv.putText(
                frame, "FIVE", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2
            )
        else:
            pass

    except:
        pass
    # displaying result
    cv.imshow("thresh", thresh)
    cv.imshow("drawing",drawing)
    cv.imshow("img",frame)


    k = cv.waitKey(30) & 0xFF  # exit if Esc is pressed
    if k == 27:
        break

cap.release()  # release the webcam
cv.destroyAllWindows()  # destroy the window
