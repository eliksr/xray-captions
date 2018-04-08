import pandas as pd
import numpy as np
import csv, random
import glob
import os
import collections

# part 1
# df = DataFrame pandas, read spreadsheet
df = pd.read_csv('MeSH/processed/openi.mesh.csv')
onehot_indexes = df['onehot_index'].unique()
print (onehot_indexes)


# part 2
# split data to training, test & validation.
tr_portion = 0.8
val_portion = 0.1
te_portion = 0.1
tr_ids = []
te_ids = []
val_ids = []
for onehot_indexi in onehot_indexes:
    dfi = df[df['onehot_index'] == onehot_indexi]
    if len(dfi) < 10:
        continue
    dfi_len_val = int(np.round(len(dfi) * 0.1))  # validation size
    dfi_len_te = int(np.round(len(dfi) * 0.1))  # test size
    dfi_len_tr = len(dfi) - int(np.round(len(dfi) * 0.2))  # training size

    tr_ids += dfi.iloc[:dfi_len_tr].id.tolist()  # takes id attr from the first ~80% with this index
    te_ids += dfi.iloc[dfi_len_tr:dfi_len_tr + dfi_len_val].id.tolist()
    val_ids += dfi.iloc[dfi_len_tr + dfi_len_val::].id.tolist()


# part 3
# slicing data
df_tr = df[df['id'].isin(tr_ids)]
df_te = df[df['id'].isin(te_ids)]
df_val = df[df['id'].isin(val_ids)]
print (len(df_tr), len(df_val), len(df_val))


# part 4
# select labels with at least 30 samples to be dataset := ds
ds = df_tr.groupby(['onehot_index']).size()[df.groupby(['onehot_index']).size()>30]
print (ds, len(ds))


# part 5
# list all the optional labels
mesh_tops = []
for i1 in ds.index:
    mesh_tops.append(df_tr.loc[df['onehot_index']==i1]['mesh_top'].unique()[0])
for mesh_topi in mesh_tops: print (mesh_topi)


# part 6
# ratios between labels in ds
ds = ds.sort_values(ascending=False)
ds_index = ds.index.tolist()
ds_value = ds.tolist()
counts_normal = ds_value[0]
counts_max = ds_value[1]
counts_min = ds_value[-1]

default_aug_ratio = 4.0
normal_max_ratio = default_aug_ratio*counts_max/counts_normal
max_min_ratio = 1.0*counts_max/counts_min
normal_min_ratio = 1.0*counts_normal/counts_min
print ('normal_max_ratio:', normal_max_ratio)
print ('max_min_ratio:', max_min_ratio)
print ('normal_min_ratio:', normal_min_ratio)


# part 7
# marked image samples to be augmented expect to the their label ratio
random.seed(123)
np.random.seed(123)

# all the normal exist
fl_pairs_tr_sa = []
# normal labels is sampled
fl_pairs_tr_sp = []
fl_pairs_te = []
fl_pairs_val = []

do_aug = False
for i2 in range(len(ds)):
    files_tr = df_tr.loc[df['onehot_index'] == ds_index[i2]]['fname']
    for filei in files_tr:
        if i2 == 0: # if normal
            do_aug = False
            fl_pairs_tr_sa.append([filei, str(i2), int(do_aug)])
            if np.random.random_sample() > normal_max_ratio:
                continue
            do_aug = False
            fl_pairs_tr_sp.append([filei, str(i2), int(do_aug)])
        else:
            for i3 in range(int(np.round(1.0 * default_aug_ratio * counts_max / ds_value[i2]))):
                if i3 == 0:
                    do_aug = False
                else:
                    do_aug = True
                fl_pairs_tr_sa.append([filei, str(i2), int(do_aug)])
                fl_pairs_tr_sp.append([filei, str(i2), int(do_aug)])

    do_aug = False
    files_te = df_te.loc[df['onehot_index'] == ds_index[i2]]['fname']
    for filei in files_te:
        fl_pairs_te.append([filei, str(i2), int(do_aug)])
    files_val = df_val.loc[df['onehot_index'] == ds_index[i2]]['fname']
    for filei in files_val:
        fl_pairs_val.append([filei, str(i2), int(do_aug)])

print (len(fl_pairs_tr_sa), len(fl_pairs_tr_sp), len(fl_pairs_te), len(fl_pairs_val))

curpath = os.path.abspath(os.curdir)
print(curpath)

# part 8
# organizing images data in several files
random.shuffle(fl_pairs_tr_sa)
random.shuffle(fl_pairs_tr_sp)
random.shuffle(fl_pairs_te)
random.shuffle(fl_pairs_val)
with open('data/iter0_im_tr_sa.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',')
    for fl_pairi in fl_pairs_tr_sa:
        csvw.writerow(fl_pairi)
with open('data/iter0_im_tr_sp.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',')
    for fl_pairi in fl_pairs_tr_sp:
        csvw.writerow(fl_pairi)
with open('data/iter0_im_te.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',')
    for fl_pairi in fl_pairs_te:
        csvw.writerow(fl_pairi)
with open('data/iter0_im_val.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',')
    for fl_pairi in fl_pairs_val:
        csvw.writerow(fl_pairi)
with open('data/iter0_im_val_0.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',')
    for fl_pairi in fl_pairs_val:
        csvw.writerow([fl_pairi[0], fl_pairi[1]])
for csvfi in glob.glob('data/iter0*.csv'): print (csvfi)


with open('data/label_names.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=' ', quotechar=' ')
    for i2 in range(len(ds)):
        label_names = df_tr.loc[df['onehot_index']==ds_index[i2]]['mesh_top']
        csvw.writerow([i2, label_names.iloc[0]])

max_label = 0

# part 10
# organizing text data in several files
fl_caps_only = []

for i2 in range(len(ds)):
    caps_tr = df_tr.loc[df['onehot_index']==ds_index[i2]]['terms_raw']
    for i4 in range(len(caps_tr)):
        capsi = caps_tr.iloc[i4]
        fl_caps_only.append(capsi)
    caps_te = df_te.loc[df['onehot_index']==ds_index[i2]]['terms_raw']
    for i4 in range(len(caps_te)):
        capsi = caps_te.iloc[i4]
        fl_caps_only.append(capsi)
    caps_val = df_val.loc[df['onehot_index']==ds_index[i2]]['terms_raw']
    for i4 in range(len(caps_val)):
        capsi = caps_val.iloc[i4]
        fl_caps_only.append(capsi)

random.shuffle(fl_caps_only)
caps_lengths_nnm = []
with open('data/iter0_caps_only.csv', 'w') as csvf:
    for fl_caps_onlyi in fl_caps_only:
        if 'normal' not in fl_caps_onlyi:
            lengthi = len(fl_caps_onlyi.split(' '))
            caps_lengths_nnm.append(lengthi)
        csvf.write(fl_caps_onlyi + '\n')
    csvf.write('_empty_')
caps_lengths_npa = np.asarray(caps_lengths_nnm)
print ('caps_lengths -- max:', caps_lengths_npa.max(), ', min:', caps_lengths_npa.min(),\
    ', mean:', caps_lengths_npa.mean(), ', std:', caps_lengths_npa.std(),\
    ', mean+3*std:', caps_lengths_npa.mean() + 3*caps_lengths_npa.std())
caps_lengths_nnmc = collections.Counter(caps_lengths_nnm) # check this operation 'Counter'
print ('-- number of elements with word length --')
for i in range(caps_lengths_npa.max()):
    print (str(i+1), ':', str(caps_lengths_nnmc[i+1]))


# part 11
# marked text samples to be augmented expect to the their label ratio
fl_imcaps_tr_sa = []
fl_imcaps_tr_sp = []
fl_imcaps_te = []
fl_imcaps_val = []

max_caps_length = 5

do_aug = False
for i2 in range(len(ds)):
    files_tr = df_tr.loc[df['onehot_index'] == ds_index[i2]]['fname']
    caps_tr = df_tr.loc[df['onehot_index'] == ds_index[i2]]['terms_raw']
    for i4 in range(len(files_tr)):
        filei = files_tr.iloc[i4]
        capsi = caps_tr.iloc[i4]

        capsi2 = capsi.split(' ')
        if len(capsi2) > max_caps_length: continue
        for i5 in range(max_caps_length - len(capsi2)): capsi2.append('_empty_')
        capsi3 = ' '.join(capsi2)

        if i2 == 0:
            do_aug = False
            fl_imcaps_tr_sa.append([filei, str(i2), int(do_aug), capsi3])
            if np.random.random_sample() > normal_max_ratio:
                continue
            do_aug = False
            fl_imcaps_tr_sp.append([filei, str(i2), int(do_aug)])
        else:
            for i3 in range(int(np.round(1.0 * default_aug_ratio * counts_max / ds_value[i2]))):
                if i3 == 0:
                    do_aug = False
                else:
                    do_aug = True
                fl_imcaps_tr_sa.append([filei, str(i2), int(do_aug), capsi3])
                fl_imcaps_tr_sp.append([filei, str(i2), int(do_aug), capsi3])

    do_aug = False
    files_te = df_te.loc[df['onehot_index'] == ds_index[i2]]['fname']
    caps_te = df_te.loc[df['onehot_index'] == ds_index[i2]]['terms_raw']
    for i4 in range(len(files_te)):
        filei = files_te.iloc[i4]
        capsi = caps_te.iloc[i4]

        capsi2 = capsi.split(' ')
        if len(capsi2) > max_caps_length: continue
        for i5 in range(max_caps_length - len(capsi2)): capsi2.append('_empty_')
        capsi3 = ' '.join(capsi2)

        fl_imcaps_te.append([filei, str(i2), int(do_aug), capsi3])
    files_val = df_val.loc[df['onehot_index'] == ds_index[i2]]['fname']
    caps_val = df_val.loc[df['onehot_index'] == ds_index[i2]]['terms_raw']
    for i4 in range(len(files_val)):
        filei = files_val.iloc[i4]
        capsi = caps_val.iloc[i4]

        capsi2 = capsi.split(' ')
        if len(capsi2) > max_caps_length: continue
        for i5 in range(max_caps_length - len(capsi2)): capsi2.append('_empty_')
        capsi3 = ' '.join(capsi2)

        fl_imcaps_val.append([filei, str(i2), int(do_aug), capsi3])

print (len(fl_imcaps_tr_sa), len(fl_imcaps_tr_sp), len(fl_imcaps_te), len(fl_imcaps_val))


# part 12
# organizing text data in several files
random.shuffle(fl_imcaps_tr_sa)
random.shuffle(fl_imcaps_tr_sp)
random.shuffle(fl_imcaps_te)
random.shuffle(fl_imcaps_val)
with open('data/iter0_imcaps_te.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',', quotechar=' ')
    for fl_pairi in fl_imcaps_te:
        csvw.writerow(fl_pairi)

with open('data/label_names.csv', 'w') as csvf:
    csvw = csv.writer(csvf, delimiter=',', quotechar=' ')
    for i2 in range(len(ds)):
        label_names = df_tr.loc[df['onehot_index']==ds_index[i2]]['mesh_top']
        csvw.writerow([i2, label_names.iloc[0]])

max_rnn_epoch = 50
inputs_dir = 'data/iter0_imcaps_trval'
if not os.path.exists(inputs_dir):
    os.makedirs(inputs_dir)
fl_paris_tr_sa_val = fl_imcaps_tr_sa + fl_imcaps_val
for i in range(max_rnn_epoch):
    inputs_dir_epochi = os.path.join(inputs_dir, 'epoch' + str(i+1))
    if not os.path.exists(inputs_dir_epochi):
        os.makedirs(inputs_dir_epochi)
    with open(os.path.join(inputs_dir_epochi, 'input.txt'), 'w') as csvf:
        csvw = csv.writer(csvf, delimiter=',', quotechar=' ')
        random.shuffle(fl_paris_tr_sa_val)
        for fl_pairi in fl_paris_tr_sa_val:
            if fl_pairi[1]==str(max_label) and np.random.random_sample() > normal_max_ratio:
                continue
            csvw.writerow(fl_pairi)
