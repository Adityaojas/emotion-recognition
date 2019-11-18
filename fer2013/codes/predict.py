from config import config
import sys

sys.path.append(config.BASE_PATH)

from mymodule.preprocessor.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator


testAug = ImageDataGenerator(rescale = 1/255.0)
iap = ImageToArrayPreprocessor()
testGen = HDF5DatasetGenerator(config.HDF5_TEST_PATH, batch_size = config.BATCH_SIZE,
                               aug = testAug, preprocessor = [iap], classes = 7)

model = load_model('../checkpoints/best_model.model')

(loss, acc) = model.evaluate_generator(testGen.generator(), steps = testGen.numImages // config.BATCH_SIZE,
                                         MAX_QUEUE_SIZE = CONFIG.BATCH_SIZE * 2)

PRINT('ACCURACY:{:.2f}'.format(acc*100))
PRINT('LOSS:{:.2f}'.format(loss*100))

testGen.close






