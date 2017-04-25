
import cv2
import imutils
from math import atan2, degrees
import numpy as np



######################## model answer ###############################

ANSWER_KEY1 = {0: 1, 1: 2, 2: 0, 3: 0, 4: 3,5:0,6:2,7:2,8:0,9:2,10:0,11:1,12:2,13:2,14:1}
ANSWER_KEY2 = {0:0,1:3,2:1,3:2,4:1,5:3,6:2,7:3,8:1,9:3,10:2,11:3,12:3,13:1,14:2}
ANSWER_KEY3 = {0:1,1:1,2:3,3:2,4:1,5:2,6:1,7:2,8:2,9:0,10:1,11:1,12:2,13:2,14:1}
offset = 1500
####################################################################

directory = "train/"

#################################################################

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

#################################################################
#############################################################
def gtc(p1):

    p = p1[offset:,:]
    gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=30, minRadius=10, maxRadius=50)

    circles = np.round(circles[0, :]).astype("int")
    c1 = circles[0]
    c2 = circles[1]
    cX, cX2, cY, cY2 = c1[0], c2[0], c1[1], c2[1]
    if cX < cX2:
        delta_x, delta_y = cX2 - cX, cY2 - cY
    else:
        delta_x, delta_y = cX - cX2, cY - cY2

    rads = atan2(delta_y, delta_x)
    angle = degrees(rads)
    image = imutils.rotate(p1,angle)

    p = p1[offset:, :]
    gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=30, minRadius=10, maxRadius=50)

    circles = np.round(circles[0, :]).astype("int")
    c1 = circles[0]
    c2 = circles[1]
    cY2 = c2[1] + offset
    return image,cY2




#########################################################
############################ modify the list of circles  ##############


def modify_this_list(circles,n):

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
    if len(sorted_questions) > 15:
        if n == 3:
            sorted_questions = sorted_questions[:15]
        elif n == 1:
            sorted_questions.pop(0)
    min_x = sorted_questions[0][0][0]

    for j in sorted_questions:
        if min_x > j[0][0]:
            min_x = j[0][0]

    for q in range(0,len(sorted_questions),1):
        o = sorted_questions[q]
        if len(o) != 4:
            y = o[0][1]
            r = o[0][2]
            new_q = [np.array([min_x,y,r]),np.array([min_x+41,y,r])
                ,np.array([min_x+82,y,r]),np.array([min_x+123,y,r])]

            sorted_questions[q] = new_q

    if len(sorted_questions) < 15:
        index = 0
        for x in range(1,len(sorted_questions),1):
            if abs(sorted_questions[x][0][1] - sorted_questions[x-1][0][1]) > 50:
                y = sorted_questions[x-1][0][1] + 42
                new_q = [np.array([min_x, y, r]), np.array([min_x + 41, y, r])
                    , np.array([min_x + 82, y, r]), np.array([min_x + 123, y, r])]
                index = x
                break

        sorted_questions.insert(index,new_q)
    return sorted_questions

#########################################################################



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
    thresh =  cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=30, minRadius=10, maxRadius=50)

    circles = np.round(circles[0, :]).astype("int")

    sorted_questions = modify_this_list(circles,n)
    correct = 0
    number_of_question = 0
    for q in sorted_questions:
        bubbled = (110,5)
        number_of_circle = 0
        list_of_bubbled = []
        count = 0
        for c in q:
            mask = np.zeros(thresh.shape, dtype="uint8")

            cv2.circle(mask, (c[0], c[1]), c[2], (255, 255, 255), -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if  total >= bubbled[0] or abs(bubbled[0] - total) < 20:
                bubbled = (total, number_of_circle)
                list_of_bubbled.append(bubbled)
                count = 1
            number_of_circle += 1
        if len(list_of_bubbled) > 1:
            max_of_total = max(list_of_bubbled)
            index = list_of_bubbled.index(max(list_of_bubbled))
            del list_of_bubbled[index]

            for ii in list_of_bubbled:
                if abs(ii[0] - max_of_total[0]) < 50:
                    count += 1
        color = (0, 0, 255)
        k = ANSWER_KEY[number_of_question]
        if k == bubbled[1] and count == 1:
            color = (0, 255, 0)
            correct += 1

        cv2.circle(p1, (q[k][0], q[k][1]), q[k][2], color,4)
        number_of_question += 1

    return correct,p1

####################################################################


#############################  Main  ###############################
directory1 = "images/"
f = open("reuslt2.csv","w")
f.write("FileName,Mark"+"\n")
read_name_of_file = open("train.csv","r")

for filename in read_name_of_file:

    filename= filename.split(",")
    filename = filename[0]
    name = str(directory + filename)
    name = name.replace("\n", "")
    name = name.replace("\r", "")
    image = cv2.imread(name)
    image,cY2 = gtc(image)
    image = image[cY2 - 785:cY2 - 80, 90:1100]
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

 ############################################################################