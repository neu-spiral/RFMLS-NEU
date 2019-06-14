import keras
import keras.models as models
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.regularizers import l2

def getHomegrownModel(slice_size=1024, classes=1000, cnn_stacks=3, fc_stacks=1, channels=128, dropout_flag=True, \
                        flt=[50, 50, 256, 80],k1=[1, 7], k2=[2, 7], batchnorm=False, dr=0.5,\
                        #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), \
                        loss='categorical_crossentropy'):
    """Original Homegrown model"""
    in_shp = [2, slice_size]
    model_nn = models.Sequential()
    model_nn.add(Reshape([1] + in_shp, input_shape=in_shp))
    model_nn.add(ZeroPadding2D((0, 2)))
    model_nn.add(Conv2D(flt[0], (k1[0], k1[1]), padding="valid", kernel_initializer="glorot_uniform", name="conv1"))
    if batchnorm:
        model_nn.add(keras.layers.BatchNormalization(momentum=0.9, name='bn_1'))
    model_nn.add(Activation('relu'))
    model_nn.add(ZeroPadding2D((0, 2)))
    model_nn.add(Conv2D(flt[1], (k2[0], k2[1]), padding="valid", kernel_initializer="glorot_uniform", name="conv2"))
    if batchnorm:
        model_nn.add(keras.layers.BatchNormalization(momentum=0.9, name='bn_2'))
    model_nn.add(Activation('relu'))
    model_nn.add(Flatten())
    model_nn.add(Dense(flt[2], kernel_initializer='he_normal', name="dense1"))
    if batchnorm:
        model_nn.add(keras.layers.BatchNormalization(momentum=0.9, name='bn_3'))
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(dr))
    model_nn.add(Dense(flt[3], kernel_initializer='he_normal', name="dense2"))
    if batchnorm:
        model_nn.add(keras.layers.BatchNormalization(momentum=0.9, name='bn_4'))
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(dr))
    model_nn.add(Dense(classes, kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), name="dense3"))
    model_nn.add(Activation('softmax'))


    model_nn.summary()




    return model_nn
