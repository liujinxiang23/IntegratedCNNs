# %%自己真正的代码
import keras
from keras.layers import Conv1D,Conv2D, Conv3D, MaxPool3D, Flatten, Dense, Reshape
from keras.layers import Dropout, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# %%
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import time
#init_notebook_mode(connected=True)
#% matplotlib inline

# %%

## GLOBAL VARIABLES
dataset = 'IP'
test_ratio = 0.70  #测试集和应用集的比例
windowSize = 25
#print('运行原始3dcnn')

# %%



def loadData(name):
    data_path = os.path.join(os.getcwd( ), '../data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path,'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'WH':  #武汉龙口
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    elif name == 'WH2':  #武汉汉川
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
    return data, labels


# %%

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


# %%

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


# %%

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# %%

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]),dtype='float32')#修改为float32
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype='float32')   #修改为float32
    #patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))#修改为float32
    #patchesLabels = np.zeros((X.shape[0] * X.shape[1]))   #修改为float32
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]  #出错
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

#MemoryError: Unable to allocate 18.0 GiB for an array with shape (257530, 25, 25, 15) and data type float64
# %%

X, y = loadData(dataset)

# %%

X.shape, y.shape

# %%

K = X.shape[2]

# %%

K = 30 if (dataset == 'IP') else 15
X, pca = applyPCA(X, numComponents=K)

# %%

X, y = createImageCubes(X, y, windowSize=windowSize)

X.shape, y.shape

# %%

Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape

# %% md

#** MOdel and traing **

# %%

Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
Xtrain.shape

# %%

ytrain = np_utils.to_categorical(ytrain)
ytrain.shape

# %%

S = windowSize
L = K
output_units = 9 if (dataset == 'PU' or dataset == 'PC' or dataset == 'WH') else 16
print(output_units)
# %%


model.summary()

# %%

adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# %%

# checkpoint
filepath = "best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# %%
start =time.time()
history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=20, callbacks=callbacks_list)
end =time.time()
print('Running time: %s Seconds'%(end-start))
# %% md

#** VALIDATION **

# %%

# load best weights
model.load_weights("best-model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# %%

Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
Xtest.shape

# %%

ytest = np_utils.to_categorical(ytest)
ytest.shape

# %%
start1 =time.time()
Y_pred_test = model.predict(Xtest)
#MemoryError: Unable to allocate 5.00 GiB for an array with shape (143180, 25, 25, 15, 1) and data type float32
end1 =time.time()
print('test time: %s Seconds'%(end1-start1))

y_pred_test = np.argmax(Y_pred_test, axis=1)

classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)


# %%

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


# %%

def reports(X_test, y_test, name):
    # start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # end = time.time()
    # print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'WH2':
        target_names = ['Strawberry','Cowpea','Soybean','Sorghum','Water spinach','Watermelon','Greens','Trees','Grass','Red roof','Gray roof','Plastic','Bare soil','Road','Bright object','Water']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100


# %%

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest, ytest, dataset)
print(confusion,'\n',"each_acc:",each_acc,'\n',"oa:",oa, '\n',"aa:",aa,'\n',"kappa",kappa)
classification = str(classification)
confusion = str(confusion)
file_name = "classification_report.txt"

with open(file_name, 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('{} each accuracy (%)'.format(each_acc))
    x_file.write('Running time'.format(end - start))
    x_file.write('\n')
    x_file.write('Test time'.format(end1 - start1))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))


# %%

def Patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch


# %%

# load the original image
X, y = loadData(dataset)

# %%

height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
numComponents = K

# %%

X, pca = applyPCA(X, numComponents=numComponents)

# %%

X = padWithZeros(X, PATCH_SIZE // 2)

# %%

# calculate the predicted image
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        target = int(y[i, j])
        if target == 0:
            continue
        else:
            image_patch = Patch(X, i, j)
            X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                               1).astype('float32')
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction + 1

# %%

ground_truth = spectral.imshow(classes=y, figsize=(7, 7))

# %%

predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))

spectral.save_rgb("predictions_ground_true.jpg", y.astype(int), colors=spectral.spy_colors)
spectral.save_rgb("predictions_20.jpg", outputs.astype(int), colors=spectral.spy_colors)






'''
