import cv2 as cv
import os
import numpy as np 
import FaceRecognition as fr 

test_img = cv.imread('C:/Users/revro/Desktop/CI Assignment/TestDataSet/EA8.png')
faces_detected, gray_img = fr.faceDetection(test_img)
print('faces_detected:', faces_detected)

# for(x, y, w, h) in faces_detected:
#     cv.rectangle(test_img, (x,y), (x+w, y+h), (255, 0, 0), thickness = 5)

# resized_img = cv.resize(test_img, (1000, 700))
# cv.imshow("Face Detection", resized_img)
# cv.waitKey(0)
# cv.destroyAllWindows

# faces, faceID = fr.labels_for_training_data('C:/Users/revro/Desktop/CI Assignment/TrainingDataSet')
# face_recognizer = fr.train_classifier(faces, faceID)
# face_recognizer.save('C:/Users/revro/Desktop/CI Assignment/trainingData.yml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/revro/Desktop/CI Assignment/trainingData.yml')

name = {0 :'Eshwer', 1 : 'Hanumanth', 2 : 'Hari'}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(roi_gray)
    print('Confidence:', confidence)
    print('Label:', label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv.resize(test_img, (1000, 700))
cv.imshow("Face Detection", resized_img)
cv.waitKey(0)
cv.destroyAllWindows