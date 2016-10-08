# Image Classification

## Overview
This exercise applies image recognition techniques to classify images in the CIFAR-10 dataset.
CIFAR-10  is an established computer-vision dataset used for object recognition. It consists of 60,000 32x32 color
images containing one of 10 object classes, with 6000 images per class.

![scene1](https://github.com/eds-uga/eatingnails-project3/blob/master/extras/cifar-10.png)

## Preprocessing
The following preprocessing was applied to each image before sending to our classifier:
- 50% chance of horizontally flipping the image
- random horizontal and/or vertical translation by 5 pixels
- ZCA whitening

## Model Description
We used a convolutional neural network as our image classifier due to the proven effectiveness of these models
for image recognition tasks. We used the following architecture/features for our network: 

## Runtime Environment
We built our final model in TensorFlow because it provided the most functionality in terms of out-of-the-box
neural network libraries. We ran our model on a single AWS G2.2xlarge instance. GPU training greatly sped up the 
time required to train our neural network. 

## Instructions for Running Model
We recommend running the following scripts on an AWS G2 instance. 
For instructions on how to setup TensorFlow on AWS, see 
[TensorFlow on AWS Instructions](https://github.com/eds-uga/eatingnails-project3/blob/master/tensorflow_on_aws.md).
Once your environment has been setup, run the following:

- python preprocessing.py \<path to X_train.txt\> \<path to X_test.txt\> \<optional: 1 to enable ZCA whitening (requires theano and scipy)\>