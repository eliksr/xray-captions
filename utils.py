import dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
from shutil import copyfile


def split_to_folders():
    path = '/Users/esror/PycharmProjects/captioning_keras/NLMCXR_dcm'
    # list_of_files = {}
    pa_lst = []
    ll_lst = []
    ap_lst = []
    non_lst = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for im_path in filenames:
            if im_path.startswith('.'):
                continue

            full_path = os.sep.join([dirpath, im_path])
            ds = dicom.read_file(full_path)
            img = ds.pixel_array.astype(np.float32)
            # if ds.PhotometricInterpretation == 'MONOCHROME1':
            #     maxI = np.amax(img)
            #     img = (2 ** int(np.ceil(np.log2(maxI - 1)))) - 1 - img

            if 'ViewPosition' in ds and ds.ViewPosition:
                if ds.ViewPosition == 'PA':
                    copyfile(full_path, os.sep.join(['data/pa', im_path]))
                elif ds.ViewPosition == 'LL':
                    copyfile(full_path, os.sep.join(['data/ll', im_path]))
                elif ds.ViewPosition == 'AP':
                    copyfile(full_path, os.sep.join(['data/ap', im_path]))
                else:
                    copyfile(full_path, os.sep.join(['data/non', im_path]))
            else:
                copyfile(full_path, os.sep.join(['data/non', im_path]))

                # plt.imshow(img, cmap=pylab.cm.binary)
                # print(ds.ViewPosition)

            # if ds.PhotometricInterpretation == 'MONOCHROME1':
            #     maxI = np.amax(img)
            #     img = (2 ** int(np.ceil(np.log2(maxI - 1)))) - 1 - img
            # # plt.imshow(img, cmap=pylab.cm.binary)
            # list_of_files[im_path] = full_path


def show_img(file):
    with open(file, 'r') as f:
        content = f.readlines()

    for path in content:
        ds = dicom.read_file(path[:-1])
        img = ds.pixel_array.astype(np.float32)
        plt.imshow(img, cmap=pylab.cm.binary)


def rewrite_mesh_file():
    with open('data/front_imgs.txt', 'r')as f:
        lines = list(map(lambda x: x.replace('\n', ''), f.readlines()))

    with open('MeSH/original/openi.mesh.top', 'r')as f:
        original_lines = f.readlines()

    res_img = []
    miss_img = []
    root = 'data/front'
    count = 0
    for line in original_lines:
        index = line.index('|')
        img_file = line.split('|')[0].split('/')[-1].replace('png', 'dcm')
        if img_file in lines:
            res_img.append(os.path.join(root, img_file) + line[index:])
            count = count +1

    with open('MeSH/original/openi.mesh.top.new', 'w') as f:
        f.writelines(res_img)

    print(len(res_img))


if __name__ == '__main__':
    # show_img('data/non_list.txt')
    # split_to_folders()
    rewrite_mesh_file()