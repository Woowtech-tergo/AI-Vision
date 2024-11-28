import numpy as np
import cv2
import time
import argparse
import Person

# Receive arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-v", "--video", help="path to the video file")
args = vars(argParser.parse_args())

# Read from webcam
if args.get("video", None) is None:
    cap = cv2.VideoCapture(0)
    time.sleep(0.25)

else: # Or read video file
    cap = cv2.VideoCapture(args["video"])

# In and out counters
leftCounter = 0
rightCounter = 0

# Set threshold
w = cap.get(3)
h = cap.get(4)
frameArea = h*w
areaTH = frameArea * 0.003

# Draw delimiting lines
leftmostLine = int(1.0/6 * w)
rightmostLine   = int(5.0/6 * w)

leftmostLimit =   int(1.0/12 * w)
rightmostLimit = int(11.0/12 * w)

leftmostLineColor = (255,0,0)
rightmostLineColor = (0,0,255)
pt1 =  [rightmostLine, 0]
pt2 =  [rightmostLine, h]
pts_L1 = np.array([pt1,pt2], np.int32)
pt3 =  [leftmostLine, 0]
pt4 =  [leftmostLine, h]
pts_L2 = np.array([pt3,pt4], np.int32)

pt5 =  [leftmostLimit, 0]
pt6 =  [leftmostLimit, h]
pts_L3 = np.array([pt5,pt6], np.int32)
pt7 =  [rightmostLimit, 0]
pt8 =  [rightmostLimit, h]
pts_L4 = np.array([pt7,pt8], np.int32)

# Create the background subtractor
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

# Create kernel for opening
kernelOp = np.ones((3,3), np.uint8)
# Create kernel for closing
kernelCl = np.ones((9,9), np.uint8)

# Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while(cap.isOpened()):

    try:
        # Read a frame
        ret, frame = cap.read()

        for per in persons:
            per.age_one()

        # Use the substractor
        fgmask = backgroundSubtractor.apply(frame)


        # Apply threshold to remove shadows
        ret, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        # Opening (erode->dilate) to remove noise
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        # Closing (dilate -> erode) to join white regions
        mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)

    except:
        # If there are no more frames to show...
        print('EOF')
        print('Left: %d' % leftCounter)
        print('Right: %d' % rightCounter)
        break


    #################
    #   CONTOURS   #
    #################

    # RETR_EXTERNAL returns only the outer contour retorna apenas o contorno externo
    # CHAIN_APPROX_SIMPLE draw contour
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #################
    #    PEOPLE     #
    #################
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > areaTH:

            # Center of mass coordinates
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            x,y,w,h = cv2.boundingRect(cnt)

            newPerson = True
            if cy in range(leftmostLimit, rightmostLimit):
                for person in persons:
                    if abs(x - person.getX()) <= w and abs(y - person.getY()) <= h:
                        # Follow detected persons until they cross the limit lines

                        newPerson = False
                        person.updateCoords(cx, cy)

                        if person.goingLeft(rightmostLine,leftmostLine) == True:
                            leftCounter += 1
                        elif person.goingRight(rightmostLine,leftmostLine) == True:
                            rightCounter += 1
                        break

                    if person.getState() == '1':
                        if person.getDir() == 'right' and person.getY() > rightmostLimit:
                            person.setDone()
                        elif person.getDir() == 'left' and person.getY() < leftmostLimit:
                            person.setDone()

                    if person.timedOut():
                        # remove i from the list of people
                        index = persons.index(person)
                        persons.pop(index)
                        del person

                if newPerson == True:
                    person = Person.Person(pid, cx, cy, max_p_age)
                    persons.append(person)
                    pid += 1

                #################
                #   DRAWINGS    #
                #################
                cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
                img = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                #cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)


    ###################
    # FRAME DETAILS  #
    ##################
    leftMsg = 'Left: ' + str(leftCounter)
    rightMsg = 'Right: ' + str(rightCounter)
    frame = cv2.polylines(frame,[pts_L1],False,rightmostLineColor,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,leftmostLineColor,thickness=2)
    frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv2.putText(frame, leftMsg ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, leftMsg ,(10,40),font,0.5,leftmostLineColor,1,cv2.LINE_AA)
    cv2.putText(frame, rightMsg ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, rightMsg ,(10,90),font,0.5,rightmostLineColor,1,cv2.LINE_AA)

    cv2.imshow('People Counting',frame)

    #Abort and exit with 'Q' or ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows
