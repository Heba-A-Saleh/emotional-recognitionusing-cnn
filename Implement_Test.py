# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 00:20:09 2018

@author: Musaoglu
"""

import cv2
import dlib 
import numpy as np
import pickle
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import linear_model
#from sklearn.neural_network import MLPClassifier
import tensorflow as tf

Sad = pickle.load(open("Sad.p","rb"))
Happy = pickle.load(open("Happy.p","rb"))
Surprise = pickle.load(open("Surprise.p","rb"))

#pickle.dump(features, open("Unknown.p","wb"))
Input_Image = cv2.imread("C:/Users/Musaoglu/Emotions_Features_Extract/Mixed Emotions/24.jpg")
Face_Detector = dlib.get_frontal_face_detector() # Detecting Face in the Image
Features_Extractions = dlib.shape_predictor("C:\\Users\\Musaoglu\\Emotions_Features_Extract\\shape_predictor_68_face_landmarks.dat") # The Landmark points for the face features
Image_Convert = cv2.cvtColor(Input_Image, cv2.COLOR_BGR2GRAY) # Convert to Gray Scale
Contrast_Image_Clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Increasing the contrast of the image
Contrast_Image = Contrast_Image_Clahe.apply(Image_Convert) # Applying


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
        
        landmarks_data= []
        for x_coord, y_coord in zip(X_Coordinates, Y_Coordinates):
            landmarks_data.append(x_coord) #adding the x-coordinates to the landmark data
            landmarks_data.append(y_coord) #adding the y-coordinates to the landmark data
            array_mean = np.asarray((Y_Coordinates_Mean,X_Coordinates_Mean)) #Converting the mean list into array 
            array_coordinates = np.asarray((y_coord,x_coord)) #Convering the coordinates list into array
            distance = np.linalg.norm(array_coordinates-array_mean) #Measuing the Normalized distance
            landmarks_data.append(int(distance)) # Adding the Normalized distance to the landmark data
        return landmarks_data # Saving the Landmark data

landmark = get_landmarks(Contrast_Image)
np.array(landmark)
    
def data():
    X = []
    Y = []
    for Happ in Happy.values():
        H = Happ
    
    for Sa in Sad.values():
        S = Sa
    
    for Sur in Surprise.values():
        Su = Sur
        
    for i in H:
        X.append(i)
        Y.append("Happy")
    
    for j in S:
        X.append(j)
        Y.append("Sad")
        
    for z in Su:
        X.append(z)
        Y.append("Surprise")
    
    return np.array(X), np.array(Y)

X,y = data()

def Classifier(x):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X)
    classifier = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=feature_columns)
    classifier.fit(x=X,y=y, batch_size=50, steps=4000)
    predictied_class = classifier.predict([x])
    return predictied_class

print (Classifier(landmark))
#for dat in Test:
#    print (Classifier(dat))
