# coding: utf-8
from keras.layers import *

from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.models import Model

shink = [4, 8, 16, 32]

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform())(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def convolutional_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.

    Returns:
      Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform())(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1',
               kernel_initializer=glorot_uniform())(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_body(x, level):
    # the main body of the resnet
    
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1',
               kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = convolutional_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    if level == 1:
        return x
    
    x = convolutional_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
    if level == 2:
        return x
    
    x = convolutional_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
    if level == 3:
        return x
    
    x = convolutional_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    return x
    
def ResNetTF(input_shape, output_shape, output_name='fc1000', weights=None, level=1, name='resnetTF'):
    if level == 0 or level > 4:
        print ("Please specify a level between 1 and 4")
        return 
    
    if input_shape[1] < 32:
        print ("Min input shape (32, 32)")
        return 
    
    x_input = Input(shape=input_shape)
    
    x = resnet_body(x_input, level)
    
    pool_size = min(7, max(1, input_shape[1]/shink[level-1]))
    if pool_size > 1:
        x = AveragePooling2D((pool_size, pool_size), name='avg_pool')(x)

    x = Flatten()(x)
    #x = Dropout(0.5, name='dropout')
    x = Dense(256, activation='relu', name='fc1')(x)
    # = Dropout(0.5, name='dropout')
    x = Dense(output_shape, activation='softmax', name=output_name,
               kernel_initializer=glorot_uniform())(x)

    # Create model.
    model = Model(x_input, x, name='resnet50')

    # load weights 
    if weights:
        print ("Adding pre-trained weights from %s" % weights)
        model.load_weights(weights, by_name=True)

    return model 