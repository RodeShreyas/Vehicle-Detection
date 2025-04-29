# Vehicle-Detection
Vehicle Detection system which will detect objects whether it is Cars or Trucks

# Aim and Objectives

## Aim
To create a real-time video of Car or Truck detection system which will detect objects based on whether it is Car or Truck.

## Objectives
➢ The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

➢ The primary objective of this project is to develop a machine learning model capable of accurately distinguishing between cars and trucks based on feature-based data.

➢ Using appropriate datasets for recognizing and interpreting data using machine learning.

➢ To show on the optical view finder of the camera module whether objects are Car or Truck.

## Abstract
➢ This project presents a machine learning and data science approach to distinguish between Cars and Trucks.

➢ An object is classified based on whether it is Car or Truck is detected by the live feed from the system’s camera.

➢ We have completed this project on jetson nano which is a very small computational device.

➢ A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

➢ One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

➢ In recent years, the ability to automatically classify vehicles has become increasingly important for applications such as traffic management, toll collection, autonomous driving, and smart city planning.

➢ Automatic classification helps traffic authorities monitor the flow of different vehicle types (cars, trucks, etc.). Trucks often move slower and cause more congestion, so real-time data allows dynamic traffic light control, lane management, and rerouting heavy vehicles to optimize road usage.

➢ Modern toll booths use automated systems to charge different rates based on vehicle type. Machine learning models can accurately classify vehicles, ensuring correct toll charges without manual checks, making the process faster and reducing human error.

## Introduction
➢ The goal of this project is to build a machine learning model that can automatically distinguish between images of cars and trucks. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

➢ Image classification tasks like this are a basic but important application of machine learning and computer vision. Such models can be useful in areas like traffic monitoring, automated toll collection, and smart city projects.

➢ Neural networks and machine learning have been used for these tasks and have obtained good results.

➢ Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Car and Truck detection as well.

Key Steps Involved:

1. Data Collection:
Gather a dataset of car and truck images.

2. Data Preprocessing:
Resize images, normalize pixel values, augment data (like rotation, flipping) to improve model generalization.

3. Training:
Train the model on the images and validate it using a separate validation set.

4. Evaluation:
Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

5. Deployment (Optional):
Deploy the model as a web app or mobile app for real-world use.

# Jetson Nano Compatibility
➢ The power of modern AI is now available for makers, learners, and embedded developers everywhere.

➢ NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

➢ Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

➢ NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

➢ In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Methodology
The Car and Truck detection system is a program that focuses on implementing real time Car and Truck detection.

It is a prototype of a new product that comprises of the main module: Car and Truck detection and then showing on view finder whether the object is Car or Truck or not.

Car and Truck Detection Module
This Module is divided into two parts:
1] Car and Truck detection
➢ Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

➢ This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.

2] Classification Detection
➢ Classification of the object based on whether it is Car and Truck or not.

➢ Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

➢ YOLOv5 was used to train and test our model for various classes like Car and Truck. We trained it for 149 epochs and achieved an accuracy of approximately 91%.

## Setup

# Installation

## Initial Setup

## Remove unwanted Applications.
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

## Create Swap file
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab

#################add line###########
/swapfile1 swap swap defaults 0 0
## Cuda Configuration
vim ~/.bashrc

#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

source ~/.bashrc

## Update a System
sudo apt-get update && sudo apt-get upgrade
## ################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
vim ~/.bashrc
sudo pip3 install pillow
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo python3 -c "import torch; print(torch.cuda.is_available())"

# Installation of torchvision.
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
# Conclusion
In this project, a machine learning model was developed to classify images into car and truck categories. By carefully preprocessing the data, applying data augmentation, and selecting an efficient classification model, we achieved good accuracy and reliable performance. The model was able to correctly handle variations in image quality and conditions. This approach shows the potential of machine learning for practical vehicle classification tasks in real-world applications.

https://github.com/user-attachments/assets/c9836a1a-5516-49db-84be-e13d300755bc

Youtube Link: https://youtu.be/L8vd-qOkqnk

