from config import config
import sys

sys.path.append(config.BASE_PATH)

from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


testAug = ImageDataGenerator(rescale = 1/255.0)
iap = ImageToArrayPreprocessor()
testGen = HDF5DatasetGenerator(config.HDF5_TEST_PATH, config.BATCH_SIZE,
                               aug = testAug, preprocessors = [iap], classes = 7)

model = load_model('../checkpoints/best_model.model')

(loss, acc) = model.evaluate_generator(testGen.generator(), steps = testGen.numImages // config.BATCH_SIZE,
                                         max_queue_size = config.BATCH_SIZE * 2)

print('ACCURACY:{:.2f}'.format(acc*100))
print('LOSS:{:.2f}'.format(loss*100))

testGen.close()
