import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 

class Patches(layers.Layer): 
    """
    Patches takes an input image and breaks it into patches of size patch_size x patch_size. Changes 
    input tensor shape to be (batch_size, # patches, patch_data) where patch data is patch_size * 
    patch_size * 3 (to conserve RGB data of each pixel in the patch) 
    *Extends tensorflow.keras.layers.Layer
        
    Attributes
    ----------
        patch_size: int 
            Length of one side of patch (patch will be patch_size by patch_size) 

    
    Methods
    -------
        call(inputs):
            Applies layer to inputs
        compute_output_shape(inputs_shape):   
            Returns the output shape of layer
    """
    def __init__(self, patch_size, **kwargs): 
        super(Patches, self).__init__( **kwargs)
        # Save initial parameters
        self.patch_size = patch_size
        
    def call(self, images): 
        # Get batch_size of tensor for reshaping of final tensor 
        batch_size = tf.shape(images)[0]
        
        # Create patches from images, will be size self.patch_size x self.patch_size. Use strides = 
        # [1,1,1,1] to not skip any section of the image 
        patches = tf.image.extract_patches(
            images = images, 
            sizes=[1,self.patch_size, self.patch_size, 1], 
            strides=[1,self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding="VALID",
        ) 
        
        # Get size of patch data, will be patch_size x patch_size * 3 (3 values for each pixel in patch) 
        patch_dims = patches.shape[-1]
        
        # Reshape patch for future layers, will be (batch_size, number_of_patches, patch_dims) 
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        
        return patches
        
    def compute_output_shape(self, input_shape):
        return (None, None, (self.patch_size**2) * 3) 