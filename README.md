# Vehicle Detection
Vehicle Detection system which will detect objects whether it is Car,Truck,Bus, or Bike.

# Aim and Objectives

## Aim
To create a real-time video of Vehicle detection system which will detect objects based on whether it is Car,Truck,Bus, or Bike.

## Objectives
➢ The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

➢ The primary objective of this project is to develop a machine learning model capable of accurately distinguishing between cars and trucks based on feature-based data.

➢ Using appropriate datasets for recognizing and interpreting data using machine learning.

➢ To show on the optical view finder of the camera module whether objects are Car,Truck,Bus, or Bike.

## Abstract
➢ This project presents a machine learning and data science approach to distinguish between Vehicles.

➢ An object is classified based on whether it is Car,Truck,Bus, or Bike. is detected by the live feed from the system’s camera.

➢ A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

➢ One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

➢ In recent years, the ability to automatically classify vehicles has become increasingly important for applications such as traffic management, toll collection, autonomous driving, and smart city planning.

➢ Automatic classification helps traffic authorities monitor the flow of different vehicle types (cars, trucks, etc.). Trucks often move slower and cause more congestion, so real-time data allows dynamic traffic light control, lane management, and rerouting heavy vehicles to optimize road usage.

➢ Modern toll booths use automated systems to charge different rates based on vehicle type. Machine learning models can accurately classify vehicles, ensuring correct toll charges without manual checks, making the process faster and reducing human error.

## Introduction
➢ The goal of this project is to build a machine learning model that can automatically distinguish between images of Car,Truck,Bus, and Bike. We are going to implement this project with Machine Learning and this project can be even run on jetson nano.

➢ Image classification tasks like this are a basic but important application of machine learning and computer vision. Such models can be useful in areas like traffic monitoring, automated toll collection, and smart city projects.

➢ Neural networks and machine learning have been used for these tasks and have obtained good results.

➢ Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Vehicle detection as well.

Key Steps Involved:

1. Data Collection:
Gather a dataset of Vehicle images from Google.

2. Data Preprocessing:
Resize images, normalize pixel values, augment data (like rotation, flipping) to improve model generalization.

3. Training:
Train the model on the images and validate it using a separate validation set.

4. Deployment (Optional):
Deploy the model as a web app or mobile app for real-world use.

# Methodology
The Vehicle detection system is a program that focuses on implementing real time Vehicle detection.

It is a prototype of a new product that comprises of the main module: Vehicle detection and then showing on view finder whether the object is Car,Truck,Bus,etc or not.

Vehicle Detection Module
This Module is divided into two parts:
1] Vehicle detection
➢ Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from google images and made our own dataset.

➢ This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.

2] Classification Detection
➢ Classification of the object based on whether it is any type of Vehicle or not.

➢ Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

➢ YOLOv5 was used to train and test our model for various classes like Car, Truck, Bus, or Bike. We trained it for 149 epochs and achieved an good accuracy.

# Download weights and Test Yolov5 Installation on USB webcam
```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```


# Vehicle Dataset Training
## We used Google Colab And Roboflow
Train our model on colab and download the weights and pass them into yolov5 folder link of project.

# Advantages
➢ Automates vehicle identification, reducing the need for manual monitoring. Saves time and effort in traffic management and surveillance systems.

➢ Improves accuracy over traditional manual or rule-based classification methods. Scalable for use in large-scale systems like toll booths, parking lots, and smart cities.

➢ Can work in real-time with proper optimization and hardware support. Adaptable to other vehicle types with minimal changes to the model.

➢ Enhances data collection for traffic analysis and planning. Cost-effective solution once deployed, especially for automated systems. Portable implementation possible on edge devices or mobile platforms.

# Applications
1. Traffic Monitoring Systems
Automatically detect and count vehicles for traffic flow analysis.

2. Toll Collection Booths
Classify vehicle types to apply correct toll charges.

3. Smart Parking Systems
Allocate parking spaces based on vehicle type (car/truck/bus/bike).

4. Environmental Monitoring
Identify heavy vehicles contributing more to emissions for regulatory purposes.

# Future Scope
➢ As we know technology is marching towards automation, so this project is one of the step towards automation.

➢ Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

➢ This project can be expanded in several meaningful directions. Future work may include extending the model to classify multiple types of vehicles beyond car, bus, etc.  enabling more detailed traffic analysis.

➢ Additionally, vehicle-type data can support urban planning, environmental monitoring, and the development of intelligent transportation policies.

➢ In future our model which can be trained and modified with just the addition of images can be very useful.
# Conclusion
In this project, a machine learning model was developed to classify images into car, bus, bike and truck categories. By carefully preprocessing the data, applying data augmentation, and selecting an efficient classification model, we achieved good accuracy and reliable performance. The model was able to correctly handle variations in image quality and conditions. This approach shows the potential of machine learning for practical vehicle classification tasks in real-world applications.

https://github.com/user-attachments/assets/c9836a1a-5516-49db-84be-e13d300755bc

Youtube Link: https://youtu.be/HbuLRC8hSqI

