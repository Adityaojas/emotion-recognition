from config import config
import sys
sys.path.append(config.BASE_PATH)

import matplotlib
matplotlib.use('Agg')

from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.callbacks.epochcheckpoint import EpochCheckpoint
from mymodule.callbacks.trainingmonitor import TrainingMonitor
from mymodule.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
from model import Model
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type = str, help = 'Path to the model checkpoint to load')
ap.add_argument('-e', '--epoch_start', type = int, default = 0, help = 'Epoch to restart training at')
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(rotation_range = 15, zoom_range = 0.15,
                              horizontal_flip = True, rescale = 1/255.0,
                              fill_mode = 'nearest')
valAug = ImageDataGenerator(rescale = 1/255.0)

iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.HDF5_TRAIN_PATH, config.BATCH_SIZE, aug = trainAug,
                                preprocessors = [iap], classes = 7)

valGen = HDF5DatasetGenerator(config.HDF5_VAL_PATH, config.BATCH_SIZE, aug = valAug,
                                preprocessors = [iap], classes = 7)

if args['model'] == None:
    
    model = Model.build(48,48,1,7)
    opt = Adam(lr = 1e-3)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

else:
    model = load_model(args['model'])
    print('Old Learning Rate:{}'.format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-4)
    print('New learning rate:{}'.format(K.get_value(model.optimizer.lr)))
    
tmPath = config.OUTPUT_PATH + 'training_monitor.png'
jsonPath = config.OUTPUT_PATH + 'history.json'
cp_path = config.CHECKPOINTS_PATH
bestmodel_path = config.CHECKPOINTS_PATH + '/best_model_2.model'

training_monitor = TrainingMonitor(tmPath, jsonPath = jsonPath, startAt = args['epoch_start'])
epoch_checkpoint = EpochCheckpoint(cp_path, every = 5, startAt = args['epoch_start'])
bestmodel = ModelCheckpoint(bestmodel_path, monitor = 'val_acc', mode = 'max',
                     save_best_only = True, verbose = 1)

callbacks = [training_monitor, epoch_checkpoint, bestmodel]

model.fit_generator(trainGen.generator(), steps_per_epoch = trainGen.numImages // config.BATCH_SIZE,
                    validation_data = valGen.generator(), validation_steps = valGen.numImages // config.BATCH_SIZE,
                    epochs = 30, callbacks = callbacks, max_queue_size = config.BATCH_SIZE*2, verbose = 1)

trainGen.close()
valGen.close()    
  




  







