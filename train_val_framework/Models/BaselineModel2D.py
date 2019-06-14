import keras
import keras.models as models
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape

def getBaselineModel2D(slice_size=64, classes=1000, cnn_stacks=3, fc_stacks=1, channels=128, dropout_flag=True, \
                        fc1=256, fc2=128, batchnorm=False, \
                        #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), \
                        loss='categorical_crossentropy'):
    model = models.Sequential()
    model.add(Conv2D(channels,(7, 2),activation='relu', padding='same', input_shape=(slice_size, 2, 1)))
    model.add(Conv2D(channels,(5, 2),activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(channels,(7, 2),activation='relu', padding='same'))
    model.add(Conv2D(channels,(5, 2),activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(channels,(7, 2),activation='relu', padding='same'))
    model.add(Conv2D(channels,(5, 2),activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(fc1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(fc2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    # optimizer = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model