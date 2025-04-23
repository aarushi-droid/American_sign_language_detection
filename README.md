🕊️ American Sign Language Detection using MediaPipe & KNN

This project implements a real-time American Sign Language (ASL) detection system using hand tracking via MediaPipe and classification using a K-Nearest Neighbors (KNN) machine learning model.

💡 Overview

The goal of this project is to detect and recognize ASL signs from a live webcam feed. MediaPipe is used to extract 21 3D hand landmark coordinates (x, y, z), resulting in a total of 63 features per detected hand. These features are then passed into a trained KNN classifier to predict the corresponding ASL sign.

📁 Project Structure

asl-sign-language-detection/
├── all_landmarks.csv       # ASL dataset with 63 hand landmark features
├── model_training.py       # Script to train and save the KNN model
├── asl_model_knn.pkl       # Trained KNN model file (auto-generated)
├── asl_predictor.py        # Real-time ASL detection using webcam
└── README.md               # Project documentation

🔧 How It Works

1. Hand Landmark Detection

Using MediaPipe's hand solution, the system detects 21 hand landmarks in real-time from webcam input.

2. Feature Extraction

Each detected hand produces 63 features (21 landmarks * 3 coordinates: x, y, z).

3. Model Training

The model_training.py script uses these features to train a KNN classifier on labeled ASL data stored in all_landmarks.csv.

4. Real-Time Prediction

The asl_predictor.py loads the trained model and classifies signs in real-time using live webcam input.

📊 Dataset

The dataset used is a custom CSV file (all_landmarks.csv) containing 63 landmark coordinates per sample, with corresponding labels indicating the ASL sign.

🚀 Getting Started

Prerequisites

Python 3.10+

pip

Installation

Install required libraries:

pip install opencv-python mediapipe scikit-learn pandas joblib numpy

Train the Model

python model_training.py

This will generate a asl_model_knn.pkl model file.

Run the Predictor

python asl_predictor.py

Use the webcam to view real-time hand tracking and sign prediction.

📈 Example Output

Real-time hand tracking

Predicted sign displayed on video frame

✨ Features

Real-time ASL sign prediction

Uses lightweight KNN instead of deep learning

Easily extendable for more gestures

⚡ Future Improvements

Add support for both hands

Create a larger dataset with more ASL signs

Integrate text-to-speech output

Build a GUI for better usability
