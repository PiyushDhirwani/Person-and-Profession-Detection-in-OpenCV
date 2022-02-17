
import cv2 as cv
import numpy as np
import dlib as dl
import face_recognition as fc
import os
from datetime import datetime as dt
 

d={'Amitabh Bachchan':'Big-B','Elon Musk':'Tesla-spaceX-owner','Shahruk Khan':'Bollywood-Superstar','Hrithik Roshan':'Super-30'}


path = 'Train'
imgs = []
clsn = []  
train = os.listdir(path)
print(train)
for cl in train:
    curImg = cv.imread(f'{path}/{cl}')
    imgs.append(curImg)
    clsn.append(os.path.splitext(cl)[0])
print(clsn)
 

def markAttendance(name):
    with open('present.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = dt.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



def findEncodings(imgs):
    encodeList = []
    for img in imgs:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
encodeListKnown = findEncodings(imgs)
 
cap = cv.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)
 
    facesCurFrame = fc.face_locations(imgS)
    encodesCurFrame = fc.face_encodings(imgS,facesCurFrame)
 
    for enfc,fcloc in zip(encodesCurFrame,facesCurFrame):
        matches = fc.compare_faces(encodeListKnown,enfc)
        faceDis = fc.face_distance(encodeListKnown,enfc)
        # print(faceDis)
        match_index=np.argmin(faceDis)
        if matches[match_index] :
            name=clsn[match_index]
            print(name, d[name])
            markAttendance(name)
    # cv.imshow('djf',img)
    # cv.waitKey(0)


