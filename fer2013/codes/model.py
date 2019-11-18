from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras import backend as K

class Model:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        
        model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape = input_shape))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))

        model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))

        model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer = 'he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(64, kernel_initializer = 'he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, kernel_initializer = 'he_normal'))
        model.add(Activation('softmax'))
        
        return(model)
        
        
        
        
        
        
        
        




