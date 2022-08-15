#! /usr/bin/python3

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer): 
    """
    PositionalEncoding Layer applies Input Embedding and Positional Encoding to initial Inputs of the Transformer
    *Extends tensorflow.keras.layers.Layer
        
    Attributes
    ----------
        sequence_length: int
            Length of the input sequence. Also used as the size of the vocabulary of the encoding
            since we are encoding the positon data of the sequence, which is a fixed sized. 
        projection_dim: int
            Dimensionality of dense projection output
    
    Methods
    -------
        call(inputs: tensor (None, sequence_length, projection_dim):
            Applies layer to inputs
        compute_output_shape(inputs_shape):   
            Returns the output shape of layer
    """
    def __init__(self, sequence_length, projection_dim, **kwargs): 
        super(PositionalEncoding, self).__init__(**kwargs) 
        # Layer to compute positional data
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=projection_dim) 
        
        # Linear project layer for input 
        self.projection = layers.Dense(units=projection_dim) 
        
        # Save initialization parameters
        self.sequence_length = sequence_length
        self.projection_dim = projection_dim
        
    def call(self, inputs):
        # Create tensor of length position
        positions = tf.range(start=0, limit=self.sequence_length, delta=1) 
        
        # Apply embedding to each position 
        encoded_positions = self.position_embeddings(positions) 
        
        # Apply linear projection to input 
        projection = self.projection(inputs) 
        
        # Combine input with positional information 
        encoding = projection + encoded_positions
        
        return encoding
        
    def compute_output_shape(self, input_shape):
        return (None, self.projection_dim)

class Encoder(layers.Layer): 
    """
    Encoder Layer applies multihead attention to inputs (using inputs as Query, Keys, and Values), as well as layer 
    normalization and dense projection according to transformer designed by Vaswani et al. (Note: Mask not implemented) 
    *Extends tensorflow.keras.layers.Layer
        
    Attributes
    ----------
        embed_dim: int      
            Dimensionality of the input embedding (conserved throughout) 
        dense_dim: int
            Dimensionality of the first dense projection hidden layer
        num_heads: int   
            Number of attention heads (for multi-head attention) 
        activation: string 
            Activation function for dense layer (default=relu) 
    
    Methods
    -------
        call(inputs):
            Applies layer to inputs
        compute_output_shape(inputs_shape):   
            Returns the output shape of layer
    """
    def __init__(self, embed_dim, dense_dim, num_heads, activation='relu', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # Attention layer with num_head heads and embed_dim output dimension
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.25)
        
        # Linear projection layers for input with 2 hidden layers, first uses dense_dim layers then transforms back to 
        # embed_dim to keep shape the same
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation=activation), layers.Dense(embed_dim),])
        
        # Normalization layer
        self.layernorm = layers.LayerNormalization()
 
        # Save initial parameters
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
    def call(self, inputs): 
        # Compute attention matrix using Multi-Head Attention (inputs is Q, K, and V) 
        # Attention(Q,K,V) = softmax([Q dot K transpose] / sqrt(dk)) * V
        attention_matrix = self.attention(inputs, inputs, attention_mask=None)
        
        # Apply attention matrix to input and normalize (Risidual Connection) 
        normalized_attention_output = self.layernorm(inputs + attention_matrix)

        # Apply linear projection (feed forward network) to normalized result of attention layer
        projection = self.dense_proj(normalized_attention_output)
        
        # Add feed forward projection results to intial inputs and normalize one more time (Risidual Connection) 
        normalized_projection = self.layernorm(normalized_attention_output + projection) 
        
        return normalized_projection
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
def BuildEncoder(sequence_length, input_dim, embed_dim, dense_dim, num_heads, model_name="transformer"): 
    """
    BuildEncoder will build a model that applies PositionalEncoding to inputs, then feeds those 
    inputs to a transformer encoder
    
    Parameters
    ----------
    sequence_length: int 
        length of the input sequence of the transformer 
    input_dim: int
        dimensionality of the input tensor
    embed_dim: int 
        output dimension of the embedding layer 
    dense_dim: int
        number of hidden units of first hidden layer of linear projection in the Encoder 
    num_heads: int 
        number of attention heads
    model_name: string 
        name of the resulting model (default="transformer") *Must be changed if using multiple transformer layers 
        
    Returns
    -------
    model: keras.Model 
        a transformer model with input layer=keras.Input with shape (sequence_length, embed_dim), and output layer=Encoder
    """
    # Input of model, tensor will be of shape (Batch_size, sequence_length, None) where None is the dimensionality of the 
    # input data     
    inputs = keras.Input(shape=(sequence_length, input_dim))
    
    # Embedding layer, will return an embedding with embed_dim dimensionality that maintains sequence_length
    embeded = PositionalEncoding(sequence_length, embed_dim)(inputs) 
    
    # Encoder layer, will maintain input_shape. Will have num_heads attention heads, and the linear projection will have 
    # 2 hidden layers, first with dense_dim units and second with embed_dim units to maintain input_shape 
    encoded = Encoder(embed_dim, dense_dim, num_heads)(embeded)
    
    # Create the model as described
    model = keras.Model(inputs, encoded, name=model_name) 
    
    return model