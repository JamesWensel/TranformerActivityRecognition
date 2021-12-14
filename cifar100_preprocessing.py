import numpy as np 
from tensorflow import keras 
from tensorflow.keras import layers 

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data() 

print("x_train shape: ", x_train.shape) 
print("y_train shape: ", y_train.shape) 
print("x_test shape: ", x_test.shape) 
print("y_test shape: ", y_test.shape) 

inp = keras.Input(shape=(32, 32, 3)) 
x = layers.Normalization()(inp)
out = layers.Resizing(64, 64)(x) 

model = keras.Model(inp, out) 

x_train = model(x_train) 
x_test = model(x_test) 

print("Reshaped x_train: ", x_train.shape) 
print("Reshaped x_test: ", x_test.shape) 

np.save("TrainImages", x_train) 
np.save("TestImages", x_test) 
np.save("TrainLabels", y_train)
np.save("TestLabels", y_test) 