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
- 20% dropout on input
- 2 convolutional layers using 3x3 filter shape, 80 feature maps each, ELU activation
- maxpool with 2x2 window
- 2 convolutional layers using 3x3 filter shape, 160 feature maps each, ELU activation
- maxpool with 2x2 window
- 2 convolutional layers using 3x3 filter shape, 320 feature maps each, ELU activation
- feedforward layer of 2000 nodes, ELU activation, dropout 50%
- softmax layer
- adam gradient descent
- orthogonal weight initialization
- batch size 100, 80000 training iterations

## Runtime Environment
We tested our model in multiple environments including Theano, Tensorflow, and DeepLearning4j.
We built our final model in TensorFlow because it provided the most functionality in terms of out-of-the-box
neural network libraries and provided the easiest implementation of multi-GPU deployment.

We provide two versions of our final model, one for single-GPU (or CPU) operation and one for multi-GPU operation.
Our single-GPU model takes approximately 6 hours to train on an AWS G2.2xLarge instance. Our multi-GPU model was
designed for a G2.8xLarge instance and is hardcoded to run on 4 GPUs. We noticed a >2x speedup when using larger
batch sizes (>500), but with our default batch size of 100, the speedup was insignificant. Therefore we recommend
running the single-GPU model instead of the multi-GPU model.

## Instructions for Running Model
We recommend running the following scripts on an AWS G2 instance. Be sure to allocate disk space to the instance
(we recommend at least 30GB). Ensure the following Python packages are installed on your instance:
- numpy
- tensorflow (For instructions on how to setup TensorFlow on AWS, see 
[TensorFlow on AWS Instructions](https://github.com/eds-uga/eatingnails-project3/blob/master/tensorflow_on_aws.md))
- sklearn
- theano (optional, required for ZCA whitening)
- scipy (optional, required for ZCA whitening)

Once your environment has been setup, download the project files and run the following:
- **generate input data:** python preprocessing.py \<path to X_train.txt\> \<path to X_test.txt\> \<optional: 1 to enable ZCA whitening (requires theano and scipy)\>
- **predict using single GPU/CPU:** python model_single_gpu.py
- **predict using 4 GPUs:** python model_multi_gpu.py

Cross validation accuracy is recorded every 100 iterations to *accuracy.txt*. 
Predictions are saved every 5000 iterations to *prediction.txt*.