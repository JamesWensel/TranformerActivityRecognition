#! /usr/bin/python3

import math
import numpy as np

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    
    def __init__(self, x, y, batch_size=32): 
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        np.random.default_rng().shuffle(self.indices)
        
    def __len__(self): 
        return math.ceil(self.x.shape[0]/self.batch_size)
    
    def __getitem__(self, idx): 
        ind = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = self.x[ind]
        batch_y = self.y[ind]
        return batch_x, batch_y
    
    def on_epoch_end(self): 
        np.random.shuffle(self.indices)
        
        
class PredictGenerator(Sequence):
    
    def __init__(self, x, batch_size=32): 
        self.x = np.asarray(x)
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        
    def __len__(self): 
        return math.ceil(self.x.shape[0]/self.batch_size)
    
    def __getitem__(self, idx): 
        ind = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = self.x[ind]
        return batch_x