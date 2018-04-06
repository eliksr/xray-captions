from os import listdir
from pickle import dump
import numpy as np
from keras.models import load_model
from data_utils import load_dicom_img

from keras.preprocessing.image import img_to_array
from data_utils import preprocessing_image
from keras.models import Model


# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = load_model('vgg_model.h5')
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_dicom_img(filename, 224)
        # convert the image pixels to a numpy array
        image = np.array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocessing_image(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
    return features


# extract features from all images
directory = 'data/front'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))