# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:33:01 2018

@author: Musaoglu
"""

import cv2
import dlib
import numpy as np


Input_Image = cv2.imread("C:/Users/Musaoglu/Emotions_Features_Extract/Mixed Emotions/26.jpg")
Face_Detector = dlib.get_frontal_face_detector() # Detecting Face in the Image
Features_Extractions = dlib.shape_predictor("C:\\Users\\Musaoglu\\Emotions_Features_Extract\\shape_predictor_68_face_landmarks.dat") # The Landmark points for the face features
Image_Convert = cv2.cvtColor(Input_Image, cv2.COLOR_BGR2GRAY) # Convert to Gray Scale
Contrast_Image_Clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Increasing the contrast of the image
Contrast_Image = Contrast_Image_Clahe.apply(Image_Convert) # Applying
Face_Detections = Face_Detector(Contrast_Image, 1) #Detect the faces in the image
for k,d in enumerate(Face_Detections): # Each Face
    shape = Features_Extractions(Contrast_Image, d) # Features Coordinates 
    for i in range(1,68): #There are 68 landmark points on each face
        cv2.circle(Input_Image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #Drawing points around each feature

def get_landmarks(image):
    detections = Face_Detector(image, 1)
    data = {}
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = Features_Extractions(image, d) #Draw Facial Landmarks with the predictor class
        
        X_Coordinates = []
        Y_Coordinates = []
        for i in range(1,68): #Getting the X and Y Coordinates from the facial landmarks points
            X_Coordinates.append(float(shape.part(i).x)) # Add the X-Coordinates to the list
            Y_Coordinates.append(float(shape.part(i).y)) # Add the Y-Coordinates to the list
            
        X_Coordinates_Mean = np.mean(X_Coordinates) #Finding the X-Coordinate of the central point
        Y_Coordinates_Mean = np.mean(Y_Coordinates) #Finding the Y-Coordinate of the central point
        
        #for x in X_Coordinates:
        #    Central_Point_X = [(x-X_Coordinates_Mean)] #The distance between each X-Coordinate and the central point
        #for y in Y_Coordinates:    
        #    Central_Point_Y = [(y-Y_Coordinates_Mean)] #The distance between each Y-Coordinate and the central point
        
        landmarks_data = [] #Landmarks Data List
        for x_coord, y_coord in zip(X_Coordinates, Y_Coordinates):
            landmarks_data.append(x_coord) #adding the x-coordinates to the landmark data
            landmarks_data.append(y_coord) #adding the y-coordinates to the landmark data
            array_mean = np.asarray((Y_Coordinates_Mean,X_Coordinates_Mean)) #Converting the mean list into array 
            array_coordinates = np.asarray((y_coord,x_coord)) #Convering the coordinates list into array
            distance = np.linalg.norm(array_coordinates-array_mean) #Measuing the Normalized distance
            landmarks_data.append(int(distance)) # Adding the Normalized distance to the landmark data
        return landmarks_data # Saving the Landmark data

        #data["landmarks_vectorised"] = landmarks_vectorised
    #if len(detections) < 1: 
    #   data["landmarks_vectorised"] = "error"
    #   return data

landmark = get_landmarks(Input_Image)
print (landmark)
#print(len(landmark))
cv2.imwrite('24_test.jpg',Input_Image)
