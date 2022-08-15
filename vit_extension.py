import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from vit_keras import vit, utils

vit_types = {'b16': vit.vit_b16, 'b32': vit.vit_b32, 'l16': vit.vit_l16, 'l32': vit.vit_l32}

class ViT(layers.Layer): 
    """
    PositionalEncoding Layer applies Input Embedding and Positional Encoding to initial Inputs of the Transformer
    *Extends tensorflow.keras.layers.Layer
        
    Attributes
    ----------
        vit_type: String
            vit implimentation to use for feature extraction
        activation: String
            activation fuction to use for training
        pretrained: bool
            use pretrained implementaion or randomly generated weights
        include_top: bool 
            include top dense layer for classification
        classes: bool
            total number of dense node in top layer

    Methods
    -------
        call(inputs: tensor (None, sequence_length, projection_dim):
            Applies layer to inputs
        compute_output_shape(inputs_shape):   
            Returns the output shape of layer
    """
    def __init__(self, vit_type, image_size, activation, pretrained, include_top, pretrained_top, classes=1024, **kwargs): 
        super(ViT, self).__init__(**kwargs) 
        
        self.classes = classes
        
        # Layer to compute positional data
        self.vision_t = vit_types[vit_type](image_size=image_size, activation=activation, pretrained=pretrained, classes=classes, include_top=include_top, pretrained_top=pretrained_top)
        
    def call(self, inputs):
        t = self.vision_t(inputs) 
        return t
        
    def compute_output_shape(self, input_shape):
        return (None, self.classes) 
