#! /usr/bin/python3

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import os
import time
import math
import psutil
import numpy as np
import tensorflow as tf

import output
from data_generator import DataGenerator
from data_generator import PredictGenerator

# Define name for function to get memory of current process
GET_MEM = psutil.Process(os.getpid()).memory_info

def FitModel(features, labels, model, model_name, OutputData=output.Output(), batch_size=128, epochs=50): 
    """
    Performs training on model with specified data and with specified settings 
    
    Arguments
    ---------
        features: numpy.array 
            Feature data to train on 
        labels: numpy.array
            Labels associated with features array 
        model: keras.Model
            Model to train with provided data
        model_name: String
            Name of model, used to save a representation of the model architecture to a file 
        OutputData: Object (default Output Object)  
            Object used to store timing, memory use, and accuracy of model
        batch_size: int (default=128)
            Size of batches of training data the model will train on at each time step  
        epochs: int (default=50)
            Number of epochs to train on the data  
    
    Returns
    -------
        model: keras.Model  
            The trained model 
        predictions: 
            Predictions made on the test data
        times: list of dictionaries
            List containing all timing data
        eval_history: 
    """

    # Add first data entry for data processing
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Processing")
    
    # Convert labels to one_hot encoding 
    oh_labels = to_categorical(labels)
    
    # Split features and labels into training and testing data with 80% training and 20% testing data
    feat_train, feat_test, lab_train, lab_test = train_test_split(features, oh_labels, test_size = 0.20, shuffle = True)
    
    # Plot model structure to files 
    PlotWholeModel(model, model_name) 
    
    # Print model architecture to console 
    model.summary() 
    
    # Create generators for data
    training_generator = DataGenerator(feat_train, lab_train, batch_size)
    testing_generator = DataGenerator(feat_test, lab_test, math.ceil(batch_size*.2))
    
    # Add final data entry for data processing
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Processing")
    
    # Add first data entry for model training
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Training")
    
    # Define callback to stop training if improvements are not being made, and provent overfitting
    early_stopping_callback = EarlyStopping(monitor = 'loss', patience = 15, mode = 'min', restore_best_weights = True)

    # Train model 
    #history = model.fit(feat_train, lab_train, batch_size=batch_size, validation_split=0.20, epochs=epochs, callbacks = [early_stopping_callback])
    history = model.fit(training_generator, epochs=epochs)
    #history = model.fit(training_generator, epochs=epochs, callbacks=[early_stopping_callback])
    
    # Add second data entry for model training
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Training")
    
    # Add first data entry for model testing
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Evaluation")
    
    # Evaluate model over testing data
    #eval_history = model.evaluate(feat_test, lab_test)
    eval_history = model.evaluate(testing_generator)
    
    # Save evaluation time for later outputing 
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Evaluation")

    # Use model for prediction and return resulting model and predictions 
    return PredictModel(feat_test, model, OutputData, history), history, eval_history
    
def PlotWholeModel(model, model_name): 
    """
    Plots the top level model architecture and recursively plots any functional submodel architectures
    All models are output to files "{model_name}.png"
    
    Arguments
    ---------
        model: keras.Model
            The model to plot 
        model_name: string
            The name of the model, used for file_name
    """
    # Plot top level model to file
    plot_model(model, to_file = model_name, show_shapes = True, show_layer_names = True)
    
    # Check each layer for functional sublayers 
    for layer in model.layers: 
        # If layer is itself a model, plot this model as well
        if layer.__class__.__name__ == 'Functional':
            # Plot internal layer with name = model_name_layer_name.png 
            PlotWholeModel(layer, model_name.split('.')[0] + '_' + layer.name + '.png')

def PredictModel(features, model, OutputData=output.Output(), history=None): 
    """
    Uses model to make predictions on the data in the feature array
    
    Arguments
    ---------
        features: numpy.array
            The data the model will be makeing predictions on 
        model: keras.Model  
            The model that will be making predictions
        OutputData: Object (default Output Object)  
            Object used to store timing, memory use, and accuracy of model
        history: object (default=None) 
            A history object containing accuracy, loss, validation accuracy, and validation loss
            for each epoch trained 
    
    Returns
    -------
        model: keras.Model  
            The trained model used by this function 
        predictions: tensorflow.tensor 
            Predictions made by the model
        OutputData: Object
            Output object containing all testing data
    """
    # Start timer for model predictions 
    process = psutil.Process(os.getpid())
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Predict")
    
    # Make predictions of feature data
    predictons = model.predict(features, verbose=1) 
    
    # Save prediction time for later outputting 
    OutputData.add_data(time=time.time(), mem=GET_MEM().rss, identifier="Predict")
    
    # Return model and predictions made by the model
    return model, predictons, OutputData
    