import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime

path = "ImagesAttendance"

images = []
classNames = []
myList = os.listdir(path)
# print(myList)

for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)

def findEncodings(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodings.append(encoding)
    return encodings

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        mydataList = f.readlines()
        namelist = []
        for line in mydataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

knownEncodingList = findEncodings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocCurr = face_recognition.face_locations(imgS)
    encodeCurr = face_recognition.face_encodings(imgS, faceLocCurr)

    for encode, faceLoc in zip(encodeCurr, faceLocCurr):
        matches = face_recognition.compare_faces(knownEncodingList, encode)
        facedis = face_recognition.face_distance(knownEncodingList, encode)

        matchindex = np.argmin(facedis)

        if(matches[matchindex]):
            name = classNames[matchindex].upper()
            # print(name)
            markAttendance(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Webcam", img)

    cv2.waitKey(1)
