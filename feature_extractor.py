from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50

import tensorflow as tf
import numpy as np
import output_data
import time

IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

SEQUENCE_LENGTH = 20

build_model_start = time.time()
inputs = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
res_model = ResNet50(include_top = False, weights = 'imagenet', pooling='avg')

outputs = layers.TimeDistributed(res_model)(inputs) 

model = keras.Model(inputs, outputs)
build_model_finish = time.time()

model.summary()

features = np.load("features.npy")

print(features.shape) 

predict_model_start = time.time()
features = model.predict(features, verbose=1) 
predict_model_finish = time.time() 

print(features.shape) 

output_data.print_time("Build Time", build_model_finish - build_model_start)
output_data.print_time("Predict Time", predict_model_finish - predict_model_start)

np.save("features_extracted.npy", features) 