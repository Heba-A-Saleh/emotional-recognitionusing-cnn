# Emotional Recognition System

This is a project for building an Emotional Recognition System using machine learning algorithms. The system is designed to classify emotions in a given set of facial images. The classification process includes recognizing seven basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.

## Dataset

I used dataset similar to FER-2013 dataset for training and testing our model. It contains 48x48-pixel grayscale images of faces, labeled with seven emotion categories. Due to confidentiality reasons, I am unable to share the dataset.

## System Architecture

The system is built using a Convolutional Neural Network (CNN) model. The CNN consists of three convolutional layers, each followed by a max-pooling layer. We used the Rectified Linear Unit (ReLU) activation function for each convolutional layer. After the convolutional layers, the model has two fully connected layers, which are connected to the output layer with softmax activation. The model was trained using the Adam optimizer and categorical cross-entropy loss function.

## Requirements

The following packages are required to run the system:

    numpy
    pandas
    opencv-python
    matplotlib
    sklearn
    IPython
    
## Project Overview

The project involves three stages: data extraction, training, and testing. The data is extracted from a pickle file saved in the feature extraction stage. Then the data is prepared into training and testing sets. The training set contains 600 data samples, while the testing set contains 200 data samples.

The next stage involves implementing different classifiers such as K-Neighbors Classifier, SGD Classifier, ExtraTree Classifier, Random Forest Classifier, Soft Voting Classifier, and Neural Networks algorithms such as Multi Layer Perceptron (MLP Classifier) and Perceptron algorithm. The performance of each classifier is measured using classification report, precision-recall plot, and confusion matrix.

The first stage involves extracting the facial features from each image to create the training data for the classifier. The dlib library can be used to extract facial landmarks from an image using a built-in data called shape_predictor_68_face landmarks, which contains 68 face landmarks that can be used to determine the coordinates of the eye, eyebrow, mouth, nose, and edge of the face.

To create the training data, three features were used for training, namely the x-coordinates, y-coordinates, and the mean distance between each point and the center of the face. The landmark points of all the images in the dataset are detected and read using the dLib library, and the data is stored in a list called image_info. The list contains data of 800 images, with 200 images for each emotion (happy, sad, surprise, and neutral), where each emotion is represented by 68 landmark points, and each point has three features.

## Data Preparation

The data set consists of 800 image face landmark features, each with a list of 201 elements representing the features extracted from each face. The features include the x and y coordinates of the landmark points, and the mean distance between the center of the face and each point, making a total of 3 features for each point. The training set contains 150 data for each of the emotions classes Happy, Sad, Surprise, and Neutral. The splitting ratio of the data to training and testing sets is 0.25.

## Scaling Data

The features are scaled to allow the performance of the classifiers to increase. Many machine learning algorithms perform better when the features data is scaled, such as Support Vector Machine, K-Neighbors, and Stochastic Gradient Descent. In Decision Tree, ExtraTree, Random Forest, and Normal Equation algorithms, scaling the data has no effect on performance.

## Classifiers Implementation

Each classifier's performance is measured by a classification report, a precision-recall plot, and a confusion matrix. The K-Neighbors Classifier gave an accuracy of 80.5%. The SGD Classifier achieved the highest accuracy, 94.9%, among the classifiers, while the Extra Tree Classifier had the lowest accuracy of 72.49%. The Random Forest Classifier had an accuracy of 80%. The Soft Voting Classifier, which combined the Random Forest Classifier, SGD Classifier, and Logistic Regression, gave an accuracy of 93%. The Neural Networks algorithms were implemented using Multi Layer Perceptron (MLP Classifier) and Perceptron algorithm.

## MLP Classifier

The MLP Classifier was used to improve the performance of the classifiers. It achieved an accuracy of 89.5%.

## Perceptron Classifier

The Perceptron algorithm gave an accuracy of 79.5%.

In conclusion, the SGD Classifier had the highest accuracy among the classifiers used. The Soft Voting Classifier gave better performance than using the Random Forest Classifier and Logistic Regression alone. The MLP Classifier was also able to improve the classifiers' performance.
