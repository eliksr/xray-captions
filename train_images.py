# CNN part
# read images with labels and aug vars
# print statistics on the data, train/test
# augmentation and normalization -> which normalization to use: mean+std, with rgb2yuv?
# train from scretch
# retrain vgg/googlnet/resnet network

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from create_dataset import get_dataset, get_dogcat_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10


def normalize_imagenet(arr):
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return arr


def cnn_model_generator(x_train, y_train, x_valid, y_valid, x_test, y_test, nb_epoch, batch_size, nb_classes):

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # model_vgg16_conv.summary()
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Create your own input format (here 3x200x200)
    input = Input(shape=(256, 256, 3), name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    model = Model(input=input, output=x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    stop = EarlyStopping(monitor='acc',
                             min_delta=0.001,
                             patience=2,
                             verbose=0,
                             mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
    #           verbose=1,
    #           validation_data=(x_valid,y_valid),
    #           class_weight='auto',
    #           callbacks=[stop, tensor_board]
    #           )

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    # im[:, :, 0] -= 103.939
    # im[:, :, 1] -= 116.779
    # im[:, :, 2] -= 123.68

    # Compute principal components required for ZCA
    datagen.fit(x_train)

    # Apply normalization
    print(x_test.shape)
    for i in range(len(x_test)):
        x_test[i] = datagen.standardize(x_test[i])

    for i in range(len(x_valid)):
        x_valid[i] = datagen.standardize(x_valid[i])

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=nb_epoch,
                        validation_data=(x_valid, y_valid),
                        callbacks=[stop, tensor_board])

    score = model.evaluate(x_test, y_test)

    return model, score


def cnn_model(x_train, y_train, x_valid, y_valid, x_test, y_test, nb_epoch, batch_size, nb_classes):

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # model_vgg16_conv.summary()
    for layer in model_vgg16_conv.layers[:25]:
        layer.trainable = False

    # Create your own input format (here 3x200x200)
    input = Input(shape=(256, 256, 3), name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    model = Model(input=input, output=x)

    # # set the first 25 layers (up to the last conv block)
    # # to non-trainable (weights will not be updated)
    # for layer in model.layers[:15]:
    #     layer.trainable = False

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    stop = EarlyStopping(monitor='acc',
                             min_delta=0.001,
                             patience=2,
                             verbose=1,
                             mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    x_train = normalize_imagenet(x_train)
    x_test = normalize_imagenet(x_test)
    x_valid = normalize_imagenet(x_valid)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_data=(x_valid,y_valid),
              class_weight='auto',
              callbacks=[stop, tensor_board]
              )

    score = model.evaluate(x_test, y_test)

    return model, score


if __name__ == '__main__':
    batch_size = 64
    nb_epoch = 50
    img_rows, img_cols = 256, 256
    nb_classes = 1
    channels = 3
    root = '/home/elik/PycharmProjects/captioning_keras/croped'

    print("Splitting data into test/ train datasets")
    # df_train = pd.read_csv('data/iter0_im_tr_sa.csv', names=['file_name', 'label', 'do_aug'])
    # df_test = pd.read_csv('data/iter0_im_te.csv', names=['file_name', 'label', 'do_aug'])
    # df_val = pd.read_csv('data/iter0_im_val.csv', names=['file_name', 'label', 'do_aug'])

    # x_train , y_train = get_dataset(df_train, img_rows)
    # x_valid, y_valid = get_dataset(df_val, img_rows)
    # x_test, y_test = get_dataset(df_test, img_rows)

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_valid = x_test
    # y_valid = y_test

    x, y = get_dogcat_dataset(img_rows)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    x_valid = x_test
    y_valid = y_test

    print("Reshaping Data")
    print("X_train Shape: ", x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, channels)

    print("X_train Shape: ", x_train.shape)
    print("X_valid Shape: ", x_valid.shape)
    print("X_test Shape: ", x_test.shape)

    print("Normalizing Data")
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_valid /= 255
    x_test /= 255

    # y_train = np_utils.to_categorical(y_train, nb_classes)
    # y_valid = np_utils.to_categorical(y_valid, nb_classes)
    # y_test = np_utils.to_categorical(y_test, nb_classes)

    print("y_train Shape: ", y_train.shape)
    print("y_train Shape: ", y_valid.shape)
    print("y_test Shape: ", y_test.shape)

    model, score = cnn_model(x_train, y_train, x_valid, y_valid, x_test, y_test, nb_epoch, batch_size, nb_classes)
    model.save('vgg_model.h5')
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    print("Predicting")
    y_pred = model.predict(x_test)

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)