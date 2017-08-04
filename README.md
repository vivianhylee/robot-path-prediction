# Robot Path Prediction
Predict HEXBUG Nano micro robot moving path in a wooden box with an obstacle path 

## Goal
Given a 1200 seconds moving robot video as training data, we build and train a model based on AI and machine learning method and use the model to predict moving robot trajectory in 10 different test scenario.

## Introduction and Algorithm Overview
* Each test scenario is a 58 seconds long video and recorded at 30 frames per second. We will predict the positions of the robot in each of the following 2 seconds (60 frames) after video subsequences (without any new measurement data).
* Robot is represented by its centroid coordinates (a pair of x, y integers) 
* We use gradient boosting regression method as our model. It takes current coordinates (x, y) and velocity from specified numbers of previous time frames as input features and outputs robot coordinates (x, y) for the next frame.


## Results
### Test 1

