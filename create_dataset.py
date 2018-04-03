import numpy as np
import pandas as pd
import os
import platform
from imgaug import augmenters as iaa
import random
import cv2
import h5py
import matplotlib.pyplot as plt
import dicom

sys = platform.system()
home = '/home/elik' if sys == 'Linux' else '/Users/esror'
root = home + '/PycharmProjects/captioning_keras/data/front'

# augmentation functions
flipper = iaa.Fliplr(1.0)
agn = iaa.AdditiveGaussianNoise(scale=0.1*255)
crop = iaa.Crop(px=(0, 10))
aug_lst = [flipper, agn, crop]


def write_pickle(data_to_write, file_path):
    pass
    # max_bytes = 2 ** 31 - 1
    # data = bytearray(obj)
    #
    # ## write
    # bytes_out = pickle.dumps(data)
    # with open(file_path, 'wb') as f_out:
    #     for idx in range(0, len(bytes_out), max_bytes):
    #         f_out.write(bytes_out[idx:idx + max_bytes])


def get_dicom_data(img_path):
    ds = dicom.read_file(img_path)
    img = ds.pixel_array.astype(np.float32)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        maxI = np.amax(img)
        img = (2 ** int(np.ceil(np.log2(maxI - 1)))) - 1 - img

    img = np.repeat(img[..., None], 3, axis=2)
    return img


def get_dataset(df, img_size, isDicom=False):
    y = []
    X = []
    for index, row in df.iterrows():
        # image = load_img(os.path.join(root, row.file_name), grayscale=True, target_size=(256, 256))
        # image = img_to_array(image)
        file_path = os.path.join(root, row.file_name)
        if os.path.exists(file_path):
            image = get_dicom_data(file_path) if isDicom else cv2.imread(file_path)  # cv2.IMREAD_GRAYSCALE
            image = cv2.resize(image, (img_size, img_size))
            image = np.array(image)

            if row.do_aug == 1:
                aug_indx = random.randint(0,2)
                aug_func = aug_lst[aug_indx]
                aug_img = aug_func.augment_image(image)
                # aug_img = preprocess_input(aug_img)

                X.append(aug_img)
                y.append(row.label)

            # image = preprocess_input(image)
            # image = image[..., np.newaxis]
            # image = np.dstack([image.astype(np.uint8)] * 3)
            X.append(image)
            y.append(row.label)

            # print('total image = ' + str(len(X)))
        else:
            print('doesn\'t exist: {}'.format(file_path))

    X = np.array(X)
    y = np.array(y)

    return X, y

    # write_pickle(X, file_path='data/' + kind + '_images_np_arr.h5')
    # write_pickle(y, file_path='data/' + kind + '_labels_np_arr.h5')
    # np.save(file='data/' + kind + '_images_np_arr.npy', arr=X)
    # np.save(file='data/' + kind + '_labels_np_arr.npy', arr=y)


def get_dogcat_dataset(img_size):
    y = []
    x = []

    for (dirpath, dirnames, filenames) in os.walk('dog_cat/train'):
        for filename in filenames:
            image = cv2.imread(os.sep.join([dirpath, filename]))  # cv2.IMREAD_GRAYSCALE
            image = cv2.resize(image, (img_size, img_size))
            x.append(image)
            if filename.startswith('cat'):
                y.append(1)
            else:
                y.append(0)
    x = np.array(x)
    y = np.array(y)

    return x, y


if __name__ == '__main__':
    # df_train = pd.read_csv('iter0_im_tr_sa.csv', names=['file_name', 'label', 'do_aug'])
    # df_test = pd.read_csv('iter0_im_te.csv', names=['file_name', 'label', 'do_aug'])
    # df_val = pd.read_csv('iter0_im_val.csv', names=['file_name', 'label', 'do_aug'])
    #
    # get_dataset(df_train)
    # get_dataset(df_test)
    # get_dataset(df_val)

    get_dogcat_dataset(256)
