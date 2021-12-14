from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

import output_data 
import tensorflow as tf
import numpy as np
import time

def FitModel(features, labels, model, model_name, times=[], batch_size=128, epochs=50): 
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
        times: List of Dictionaries 
            Contains time data for timeing model performance (default=[]) 
        batch_size: int 
            Size of batches of training data the model will train on at each time step (default=128) 
        epochs: int 
            Number of epochs to train on the data (default=50) 
    
    Returns
    -------
        model: keras.Model  
            The trained model 
        predictions: 
            Predictions made on the test data
    """
    
    # Convert labels to one_hot encoding 
    oh_labels = to_categorical(labels)
    
    # Split features and labels into training and testing data with 80% training and 20% testing data
    feat_train, feat_test, lab_train, lab_test = train_test_split(features, oh_labels, test_size = 0.20, shuffle = True)
    
    # Plot model structure to files 
    PlotWholeModel(model, model_name) 
    
    # Print model architecture to console 
    model.summary() 
    
    # Define callback to stop training if improvements are not being made, and provent overfitting
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

    # Start timer for model training 
    start_time = time.time() 
    
    # Train model 
    history = model.fit(feat_train, lab_train, batch_size=batch_size, validation_split=0.20, epochs=epochs, callbacks = [early_stopping_callback])
    
    # Save training time for later outputing
    times.append({'Name': 'Train', 
                  'Value': time.time() - start_time})

    # Start timer for model evalution 
    start_time = time.time()
    
    # Evaluate model over testing data
    eval_history = model.evaluate(feat_test, lab_test)
    
    # Save evaluation time for later outputing 
    times.append({'Name': 'Evaluate', 
                  'Value': time.time() - start_time})

    # Use model for prediction and return resulting model and predictions 
    return PredictModel(feat_test, model, times, history) 
    
def PlotWholeModel(model, model_name): 
    """
    Plots the top level model architecture and any functional submodel architectures to files 
    
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

def PredictModel(features, model, times=[], history=None): 
    """
    Uses the model to make predictions on the data in the feature array
    
    Arguments
    ---------
        features: numpy.array
            The data the model will be makeing predictions on 
        model: keras.Model  
            The model that will be making predictions
        times: List of Dictionaries
            Contains time data for timeing model performance (default=[])
        history: object 
            A history object containing accuracy, loss, validation accuracy, and validation loss
            for each epoch trained (default=None) 
    
    Returns
    -------
        model: keras.Model  
            The trained model used by this function 
        predictions: tensorflow.tensor 
            Predictions made by the model
    """
    # Start timer for model predictions 
    start_time = time.time() 
    
    # Make predictions of feature data
    predictons = model.predict(features, verbose=1) 
    
    # Save prediction time for later outputting 
    times.append({'Name': 'Predict', 
                  'Value': time.time() - start_time}) 
    
    # Output all stored times and model history
    OutputModelResults(times, history) 
    
    # Return model and predictions made by the model
    return model, predictons

def OutputModelResults(times, history=None): 
    """
    Calls output_data.py functions to output stored data from training and predicting
    
    Arguments
    ---------
        times: List of Dictionaries
            Contains time data for timeing model performance
        history: object 
            A history object containing accuracy, loss, validation accuracy, and validation loss
            for each epoch trained (default=None) 
    """
    for current_time in times: 
        output_data.print_time(current_time['Name'], current_time['Value'])
    
    if history is not None: 
        output_data.plot_metric(history)