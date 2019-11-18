import sys
from config import config

sys.path.append(config.BASE_PATH)

from mymodule.io.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np

fer_csv = open(config.CSV_PATH)
fer_csv.__next__()

trainImgs, trainLbls = [], []
valImgs, valLbls = [], []
testImgs, testLbls = [], []

for row in fer_csv:
    label, image, usage = row.strip().split(',')
    
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))
    
    if usage == "Training":
        trainImgs.append(image)
        trainLbls.append(int(label))
         
    elif usage == "PrivateTest":
        valImgs.append(image)
        valLbls.append(int(label))
    
    else:
        testImgs.append(image)
        testLbls.append(int(label))
    
datasets = [(trainImgs, trainLbls, config.HDF5_TRAIN_PATH),
            (valImgs, valLbls, config.HDF5_VAL_PATH),
            (testImgs, testLbls, config.HDF5_TEST_PATH)]


for (images, labels, out_path) in datasets:
    writer = HDF5DatasetWriter((len(images), 48, 48), out_path)
    for (image, label) in zip(images, labels):
        writer.add([image], [label])
        
    writer.close()
    
f.close()
    
        
    
    
    
    
    
    
    
    
    
    




