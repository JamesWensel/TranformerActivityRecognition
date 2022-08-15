#! /usr/bin/python3

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical

import os
import psutil
import time
import numpy as np 
import tensorflow as tf

# Personal File imports
import output
import build_models
from data_generator import DataGenerator
from data_generator import PredictGenerator

OutputData = output.Output(log_level=3)

# Dataset to load images from
DIR = "One Video"

# Get current process object and print initial memory usage
process = psutil.Process(os.getpid())
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Runtime")

# Record time and memory before loading arrays
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Load Features")

# Load features data to make predictions on
features = np.load(os.path.join(DIR,"features.npy"))
labels = to_categorical(np.load(os.path.join(DIR,"labels.npy")))

# Record time and print memory when loading finishes
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Load Features")

# Get shape of features array to determine model input shape
# Should be in the form (Num Videos, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3) type uint8
feat_shape = features.shape
print(f"Feature Shape: {feat_shape}")

# Image Dimensions
IMAGE_HEIGHT, IMAGE_WIDTH = feat_shape[2], feat_shape[3]

# Number of images per sample
SEQUENCE_LENGTH = feat_shape[1]

# Record time and print the memory used when model building starts
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Build Model")

# Create Time Distributed Vit model to apply same weights to all SEQUENCE_LENGTH images at the same time.
# We do not include top because this layer will be passed directly to an LSTM or transformer, so 
# we do not need the final output dense layer.
inputs = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
vit_model = build_models.Vision_Transformer_Model('l16', SEQUENCE_LENGTH, IMAGE_HEIGHT)
outputs = vit_model(inputs) 

# Build the model
model = keras.Model(inputs, outputs)

# Record time and print memory used when model building finishes
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Build Model")

# Show summary of modle
model.summary()

# Record time and print memory when model begins predicting
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Predicting")

# Make predictions on feature data using data generator (save tf from loading all data at once)
feature_generator = PredictGenerator(features, 4)
features = model.predict(feature_generator, verbose=1) 

# Record time and print memory when model finishes prediciting
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Predicting")

# Show changes from ResNet50
print("Predicted Features Shape: {}\n".format(features.shape)) 

# Record final data and write out saved data
OutputData.add_data(time=time.time(), mem=process.memory_info().rss, identifier="Runtime", final=True, filename=os.path.join(DIR, f"Vit_{feat_shape[2]}.csv"))

# Save extracted features 
np.save(os.path.join(DIR, "ViT_features_extracted.npy"), features)