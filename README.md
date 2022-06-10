# Automatic Braking and Speed Control System using Deep Neural-Network.

### Introduction

    Safety is a necessary part of man’s life. According to the report of NCRB of India 3,54,796 cases of road accidents reported during 2020 in which more than 60% of road accidents were caused due to over-speeding. In many road accident cases, a major cause of the accident is the driver distraction and failure to react in time or negligence of the driver or because of failure of braking system to stop the vehicle in time. The number of accidents and the effects of collision can be minimized by reducing the total stopping distance by the developments of automatic braking systems which has led to significant safety in driving. This can be done by systems like Automatic Braking Systems which can be useful as well as helpful. We make use of deep neural network for the efficient and effective speed control and braking.

### Objectives

    The purpose of Automatic Braking and Speed Control (ABSC) System is to develop an automated control system that would maintain a safe driving distance from obstacles while driving. This project focuses on developing control system based on Deep Neural Network for speed control of vehicle to curb road accidents and effectively assure safety and stress-free driving.

### Methodology

    Automatic braking and speed control System (ABSC) basically controls the speed of the vehicle by continuously feeding the driving atmosphere to the pre-trained deep neural network as digital image captured by the camera sensor​. The Neural Network predicts the desired speed for that instant. If the predicted speed is less than actual speed of the vehicle, the vehicle’s embedded unit automatically alerts the driver to reduce the speed and waits for the user response. If it doesn’t get any input from the user , the controller will calculate the change in speed between previous and current instants and correspondingly the acceleration. The controller will send signal to the actuator to move the accelerate peddle to attain speed predicted by the NN.

### Steps involved in the process

    <ol>
    <li> Collect the environment data using Camera sensor </li>
    <li> Pre-processed the collected image </li>
    <li> Feed pre-processed image to the AI model </li>
    <li> Maintain / reduce / increase the speed as predicted by the model </li>
    </ol>

### Model architecture

    A CNN typically has two layers: a convolutional layer and a fully connected layer.
    The network architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 4 fully connected layers​. The network has about 27 million connections and 250 thousand parameters. The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided.convolution with a 3×3 kernel size in the final two convolutional layers. Each layer is introduced with non-linearity with ReLU activation function and  the output layer consists of log SoftMax function for multi-class classification. The output will be any one of the range between 0-5, 5-15, 15-25, 25-35, more than 35.
