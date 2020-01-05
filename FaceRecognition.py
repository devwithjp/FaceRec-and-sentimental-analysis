import cv2 as cv
import os
import numpy as np 

def faceDetection(test_img):
    gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    face_haar_cascade = cv.CascadeClassifier('C:/Users/revro/Desktop/CI Assignment/HaarCascade/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    
    return faces, gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path, subDirNames, fileNames in os.walk(directory):
        for fileName in fileNames:
            if fileName.startswith("."):
                print("Skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, fileName)
            print("img_path:", img_path)
            print('id:', id)
            test_img = cv.imread(img_path)
            if test_img is None:
                print('Image not loaded properly')
                continue
            faces_rect, gray_img = faceDetection(test_img)
            if len(faces_rect) != 1: #To allow classifier to train with images with one face
                continue
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y+h, x:x+w]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID

def train_classifier(faces, faceID):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def draw_rect(test_img, face):
    (x,y,w,h) = face
    cv.rectangle(test_img, (x,y), (x+w, y+h), (255, 0, 0), thickness = 5)

def put_text(test_img, text, x, y):
    cv.putText(test_img, text, (x,y), cv.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2)