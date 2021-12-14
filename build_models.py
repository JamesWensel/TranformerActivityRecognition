from tensorflow import keras 
from tensorflow.keras import layers

import image_transformation
import transformer
import tensorflow as tf

def ResNet_Model(sequence_length, image_height, image_width): 
    """
    Creates a Time Distributed ResNet50 Model for feature extraction 
    
    Arguments 
    ---------
        sequence_length: int
            The length of the sequence of images (highest dimension of input tensor) (in our case, number of frames used) 
        image_height: int
            Height of the images in pixels 
        image_width: int
            Width of the images in pixels 
            
    Returns
    -------
        model: keras.Model 
            Time Distributed ResNet50 Model according to input specifications 
    """
    # Define input layer of model 
    inputs = keras.Input(shape=(sequence_length, image_height, image_width, 3))
    
    # Create ResNet50 layer with imagenet weights preloaded and global average pooling. 
    # Do not include top because data from this model is fed into future layers 
    res_model = keras.applications.ResNet50(include_top = False, weights = 'imagenet', pooling='avg')
    
    # Create output layer by applying ResNet50 over all timesteps of the input 
    outputs = layers.TimeDistributed(res_model)(inputs) 

    # Create model with premade layers
    model = keras.Model(inputs, outputs, name="ResNet_Model")
    
    # Compile the created model with loss function, optimizer, and metrics to use when training
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"]) 
    
    return model

def LSTM_Model(sequence_length, input_size, LSTM_layers, LSTM_units, categories):
    """
    Creates an LSTM Model with dropout layers after each LSTM layer, and ending in a Dense prediction layer
    
    Arguments
    ---------
        sequence_length: int 
            The length of the sequence of images (highest dimension of input tensor) (in our case, number of frames used)
        input_size: int 
            Size of the lowest dimension of the input data (amount of data at each time step) 
        LSTM_layers: int
            Number of LSTM layers to include in the model
        LSTM_units: int
            Number of units for the final LSTM layer, each preceding layer will have 2 times as many layers as the layer 
            immediately after it
        categories: int
            Number of classification categories (and number of units in Dense prediction layer) 
    
    Returns
    -------
        model: keras.Model
            LSTM Model according to input specifications
    """
    # Define input layer of model
    inputs = keras.Input(shape=(sequence_length, input_size))
    
    # Calculate number of LSTM_units at highest layer if there are more than 1 layer
    if LSTM_layers > 1: 
        LSTM_units = LSTM_units * (2 ** (LSTM_layers - 1)) #2^(LSTM_layers - 1) will equal the power of two equivalent to the top layer
        
    # Create first LSTM layer, return_sequences will be true when more layers are needed 
    lstm = layers.LSTM(LSTM_units, return_sequences = (LSTM_layers > 1))(inputs) 
    
    # Add first dropout layer 
    lstm = layers.Dropout(0.25)(lstm) 

    # Add all remaining LSTM and dropout layers 
    for i in range(LSTM_layers-1): 
        # Calculates the number of units needed for current layer, and only returns the whole sequence if it is not the last layer 
        lstm = layers.LSTM((LSTM_units // (2 ** (i+1))), return_sequences = (i != LSTM_layers - 2))(lstm) 
        lstm = layers.Dropout(0.25)(lstm) 
    
    # Creates final layer of the model, a dense prediction layer
    outputs = layers.Dense(categories, activation = 'softmax')(lstm)
    
    # Create model with input and output as defined above 
    model = keras.Model(inputs, outputs, name="LSTM_Model") 

    # Compile the created model with loss function, optimizer, and metrics to use when training
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"]) 
    
    return model

def VisionTransformer_Model(image_size, patch_width, transformer_layers, feature_embedding_dim, dense_dim, num_attention_heads, categories): 
    """
    Creates a Vision Transformer Model by first splitting input images into patches, linearly transforms them, then feeds them into 6 transformer layers 
    
    Arguments: 
        image_size: int 
            Size in pixels of one dimension of the input images 
        patch_width: int
            Size in pixels of one dimension of the patches the image will be split into 
        transformer_layers: int
            Number of Transformer Layers to add to the model
        feature_embedding_dim: int
            Dimensionality of the output of the output of the embedding layer of the transformer (will be output dimensionality of the transformer) 
        dense_dim: int
            Dimensionality of the intermediate hidden dense layer in the encoder
        num_attention_heads: int
            Number of heads of attention used in the multi-headed attention section of the transformer 
        categories: int 
            Number of classification categories possible in the output (also number of units of final dense prediction layer 
            
    Returns
    -------
        model: keras.Model 
            Vision Transformer Model accoring to input specifications
    """
    
    # Define internal parameters relating to image patches 
    num_patches = (image_size // patch_width) ** 2
    patch_size = (patch_width ** 2) * 3
    
    # Define input layer of model
    inputs = keras.Input(shape=(image_size,image_size,3))
    
    # Generate patches from each image input 
    patches = image_transformation.Patches(patch_width)(inputs) 

    # Create transformer model and apply it to image patches
    transformer = Transformer_Model(patch_size, transformer_layers, num_patches, feature_embedding_dim, dense_dim, num_attention_heads, categories)  
    outputs = transformer(patches)

    # Create model with input and output as defined above  
    model = keras.Model(inputs, outputs, name="VisionTransformer_Model")

    # Compile the created model with loss function, optimizer, and metrics to use when training
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
     
    return model

def Transformer_Model(input_size, transformer_layers, sequence_length, feature_embedding_dim, dense_dim, num_attention_heads, categories):     
    """
    Creates a Transformer Model by first applying a positional embedding to the input, then passing the combined tensor to an encoding layer
    that applies multi-headed self attention to make predctions. 
    
    Arguments 
    ---------
        input_size: int
            Size of the last dimension of the input tensor 
        transformer_layers: 
            Number of transformer layers to add to the model 
        sequence_length: int
            Length of the input sequence of the model 
        feature_embedding_dim: int
            Dimensionality of the output of the embedding layer of the transformer (will be output dimensionality of the transformer) 
        dense_dim: int
            Dimensionality of the intermediate hidden dense layer in the encoder
        num_attention_heads: int
            Number of heads of attention used in the multi-headed attention section of the transformer 
        categories: int 
            Number of classification categories possible in the output (also number of units of final dense prediction layer 
    
    Returns
    -------
        model: keras.Model 
            Transformer Model according to input specifications
    """
    # Define input layer of model
    inputs = keras.Input(shape=(sequence_length, input_size))
    
    # Create first transformer layer with input values = model input
    trans = transformer.BuildEncoder(sequence_length, input_size, feature_embedding_dim, dense_dim, num_attention_heads, "transformer")(inputs) 
    
    # Create remaining transformer layers with input values = transformer outputs 
    for i in range(transformer_layers-1):
        trans = transformer.BuildEncoder(sequence_length, feature_embedding_dim, feature_embedding_dim, dense_dim, num_attention_heads, "transformer" + str(i+2))(trans) 

    # Can use pooling instead of flatten for quicker computation 
    #pool = layers.GlobalMaxPooling1D()(trans)
    
    # Flatten final output for dense layer 
    pool = layers.Flatten()(trans) 
    
    # Add dropout layer 
    dropout = layers.Dropout(0.25)(pool)
    
    # Creates final layer of the model, a dense prediction layer
    outputs = layers.Dense(categories, activation="softmax")(dropout)
    
    # Create model with input and output as defined above 
    model = keras.Model(inputs, outputs, name="Transformer_Model") 

    # Compile the created model with loss function, optimizer, and metrics to use when training
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

    return model