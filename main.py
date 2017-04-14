############################## MCQ Corrector ##################################

from imutils import contours
import cv2
import imutils
import os
from math import atan2, degrees, pi
import numpy as np



######################## model answer ###############################

ANSWER_KEY1 = {0: 1, 1: 2, 2: 0, 3: 0, 4: 3,5:0,6:2,7:2,8:0,9:2,10:0,11:1,12:2,13:2,14:1}
ANSWER_KEY2 = {0:0,1:3,2:1,3:2,4:1,5:3,6:2,7:3,8:1,9:3,10:2,11:3,12:3,13:1,14:2}
ANSWER_KEY3 = {0:1,1:1,2:3,3:2,4:1,5:2,6:1,7:2,8:2,9:0,10:1,11:1,12:2,13:2,14:1}

####################################################################33

directory = "test/"
directory1 = "images/"

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

count_to_try = 0
######################   rotate and crop questions part   ###############################

def rotate_crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    circles = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            circles.append(c)
    circles = contours.sort_contours(circles, method="top-to-bottom")[0]
    v = [circles[len(circles) - 1], circles[len(circles) - 2]]
    cv2.drawContours(image, v, -1, 255, -1)
    M = cv2.moments(v[0])
    M1 = cv2.moments(v[1])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cX2 = int(M1["m10"] / M1["m00"])
    cY2 = int(M1["m01"] / M1["m00"])
    if cX < cX2:
        delta_x, delta_y = cX2 - cX, cY2 - cY
    else:
        delta_x, delta_y = cX - cX2, cY - cY2

    rads = atan2(delta_y, delta_x)
    angle = degrees(rads)
    image = imutils.rotate(image, angle)
    image = image[cY2 - 770:cY2 - 80, 90:1100]
    return image

##########################################################################


############################ modify the list of circles  ##################################

def modify_this_list(circles):

    c = sorted(circles, key=lambda tup: tup[0])
    min_x = c[0][0]
    circles = sorted(circles, key=lambda tup: tup[1])
    questions = []
    work = [circles[0]]
    for i in range(1, len(circles), 1):
        prev_circle = circles[i - 1]
        curruent_circle = circles[i]
        if abs(curruent_circle[1] - prev_circle[1]) <= 5:
            work.append(curruent_circle)
        else:
            questions.append(work)
            work = [curruent_circle]
            i += 1
    questions.append(work)
    sorted_questions = []

    for o in questions:
        w = sorted(o, key=lambda tup: tup[0])
        sorted_questions.append(w)

    for q in range(0,len(sorted_questions),1):
        o = sorted_questions[q]
        if len(o) != 4:
            y = o[0][1]
            r = o[0][2]
            new_q = [np.array([min_x,y,r]),np.array([min_x+41,y,r])
                ,np.array([min_x+82,y,r]),np.array([min_x+123,y,r])]

            sorted_questions[q] = new_q
    if len(sorted_questions) > 15:
        sorted_questions = sorted_questions[:15]

    return sorted_questions

##################################################################################

###########################  correct  ###################################

def correct_this_page(p1,n):
    ANSWER_KEY ={}
    if n == 1:
        ANSWER_KEY = ANSWER_KEY1
    elif n == 2:
        ANSWER_KEY = ANSWER_KEY2
    else:
        ANSWER_KEY = ANSWER_KEY3
    gray = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=30, minRadius=10, maxRadius=50)
    circles = np.round(circles[0, :]).astype("int")

    sorted_questions = modify_this_list(circles)


    correct = 0
    number_of_question = 0
    for q in sorted_questions:
        bubbled = None
        number_of_circle = 0
        for c in q:
            mask = np.zeros(thresh.shape, dtype="uint8")

            cv2.circle(mask, (c[0], c[1]), c[2], (255, 255, 255), -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, number_of_circle)
            number_of_circle += 1
        color = (0, 0, 255)
        k = ANSWER_KEY[number_of_question]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        cv2.circle(p1, (q[k][0], q[k][1]), q[k][2], color,4)
        number_of_question += 1
    return correct,p1
############################################################################

#############################  Main  ###############################
f = open("reuslt.csv","w")
read_name_of_file = open("test.csv","r")
for filename in read_name_of_file:
    name = str(directory + filename)
    name = name.replace("\n", "")
    name = name.replace("\r", "")
    image = cv2.imread(name)
    image = rotate_crop_image(image)
    p1 = image[:, 0:350]
    p2 = image[:, 380:680]
    p3 = image[:, 720:]
    correct1, p1 = correct_this_page(p1, 1)
    correct2, p2 = correct_this_page(p2, 2)
    correct3, p3 = correct_this_page(p3, 3)
    data = filename.replace("\n", "").replace("\r", "") + "," + str((correct1 + correct2 + correct3))
    f.write(data + "\n")


cv2.waitKey(0)
cv2.destroyAllWindows()



#############################################################################################





















