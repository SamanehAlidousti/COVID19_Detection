# COVID-19 Detection 
This is a COVID-19 Detector using chest X-ray images.

## Objectives
In this project, I used the pre-trained Resnet-18 model which is a Convolutional Neural Network (CNN) that has been trained on the ImageNet dataset. This dataset has 100 classes, but my dataset only has three classes: normal, viral, and covid-19. To implement this model, I adopted the last fully connected layer to 3 (number of my classes). 

## Requirements
I run this project on Jupyter Notebook.

## Dataset

The dataset can be found in Kaggle [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) which includes COVID-19, normal, and other respiratory infections is being provided in several phases. 
![Picture2](https://github.com/SamanehAlidousti/COVID19_Detection/assets/107434108/0cdf8ef7-8400-4966-aceb-8d48acef3f21)


In this project, I used the second release of the dataset with a total number of 15,063 images, including 3,586 COVID-19 cases, 10,162 normal, and 1,315 viral cases of pneumonia. The number of classes is imbalanced which could bias the behavior of the model. To address this, I used the Data Augmentation technique to increase the number of COVID-19 cases. 


## Overview

*Resnet-18
ResNet-18 is a convolutional neural network (CNN) architecture that belongs to the ResNet (Residual Network) family. The key innovation of ResNet is the introduction of residual blocks, which contain skip connections or shortcuts that allow the network to learn residual mappings. This helps in addressing the vanishing gradient problem, making it easier to train very deep neural networks. 
ResNet-18 specifically is one variant of the ResNet architecture, and its depth is 18 layers. The architecture consists of several residual blocks, and the general structure of ResNet-18 is as follows:

Initial Convolutional Layer: 7x7 convolution with 64 filters

batch normalization and ReLU activation.

Max Pooling Layer: 3x3 max pooling with stride 2.

Residual Blocks: Four sets of residual blocks, each containing two convolutional layers with batch normalization and ReLU activation.

Global Average Pooling: Average pooling layer that pools the spatial dimensions to obtain a fixed-size output.

Fully Connected Layer: A fully connected layer for the final classification.

ResNet-18 has been widely used as a baseline model for various image classification tasks due to its relatively compact architecture compared to deeper ResNet variants like ResNet-34, ResNet-50, and so on. The number in the ResNet-X naming convention represents the total number of convolutional layers in the network.

 ![Picture1](https://github.com/SamanehAlidousti/COVID19_Detection/assets/107434108/b9242ebe-af42-4bc7-8228-2782d0612c60)


In this project, I used Transfer learning method to train the model because of the limited training dataset which resulted in almost 98 percent accuracy. However, this approach could not be used in a real-world situation.

