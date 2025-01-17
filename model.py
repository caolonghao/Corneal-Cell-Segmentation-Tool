import argparse
import os

# keras 3.0 version
import os
# os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow.keras.layers as layers 
import tensorflow.keras.backend as K 
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam, Nadam

K.clear_session()  # For easy reset of notebook state.
   

##############################################################################
#  BLOCKS
##############################################################################

def dense_block(x, blocks, growth_rate, name, data_format='channels_last'):
    """A dense block.
    # Arguments
        x:          input tensor.
        blocks:     integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers (output maps).
        name:       string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
      x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1), 
                     data_format=data_format)
    return x



def conv_block(x, growth_rate, name, data_format='channels_last'):
    """ A building block for a dense block.
    # Arguments
        x:          input tensor.
        growth_rate: float, growth rate at dense layers.
        name:       string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 1 if data_format=='channels_first' else 3
    x1 = layers.Conv2D(growth_rate, 4, activation=None, padding='same',  
                       use_bias=False, data_format=data_format, 
                       name=name + '_conv')(x)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   renorm=True, name=name + '_bn')(x1)  
    x1 = layers.Activation('elu')(x1)
    x1 = layers.Dropout(rate=0.2, name=name + '_drop')(x1)
    x  = layers.Concatenate(axis=bn_axis, name=name + '_conc')([x,x1])    
    return x



def down_block(x, out_channels, name, data_format='channels_last'):
    """ The downsampling block, which contains a feature reduction layer, 
    (Conv1x1) + BRN + ELU, and a average pooling layer.
    # Arguments
        x:            input tensor.
        out_channels: float, number of output channels.
        name:         string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 1 if data_format=='channels_first' else 3
    x = layers.Conv2D(out_channels, 1, activation=None, padding='same',  
                      use_bias=False, data_format=data_format, 
                      name=name + '_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  renorm=True, name=name + '_bn')(x)  
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D(2, padding='same', 
                                data_format=data_format)(x)  
    return x


def upsampling_block(x, out_channels, name, data_format='channels_last'):
    """ The upsampling block, which contains a transpose convolution.
    # Arguments
        x:            input tensor.
        out_channels: float, number of output channels.
        name:         string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 1 if data_format=='channels_first' else 3
    x = layers.Conv2DTranspose(out_channels, 4, strides=(2, 2), 
                               activation=None, padding='same',  
                               use_bias=False, data_format=data_format, 
                               name=name + '_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  renorm=True, name=name + '_bn')(x) 
    x = layers.Activation('elu')(x)
    return x


##############################################################################
# CORE OF THE NETWORK
##############################################################################

def get_unet(image_shape, blocks, growth_rate, learning_rate, weights=None,
             data_format='channels_last'):
    """ The core of the Dense U-Net.
    # Arguments
        image_shape:  array, ints, shape of the image.
        blocks:       array, ints, the number of dense blocks for each 
                      resolution block.
        growth_rate:  float, growth rate (feature maps) at the dense layers.
        learning_rate:           float, learning rate of the optimizer.
        weights:      string, name of the h5 file with the weights (optional)
        data_format:  string to indicate whether channels go first or last. 
                      NOTE: The code only works with channels_last for now.
    # Returns the model
    """

    bn_axis = 1 if data_format=='channels_first' else 3
    comprs = [int(x*growth_rate*0.50) for x in blocks]

    inputs = layers.Input(shape=image_shape)
    
    #-------------------------------------------- Dense Block 01    
    x1  = dense_block(inputs, blocks[0], growth_rate, name='block_1', 
                      data_format=data_format)    
    x1d = down_block(x1, comprs[0], name='down_1', 
                     data_format=data_format)   
    
    #-------------------------------------------- Dense Block 02
    x2i = layers.AveragePooling2D(2, padding='same',
                                  data_format=data_format)(inputs)
    x2c = layers.Concatenate(axis=bn_axis)([x2i,x1d])
    x2  = dense_block(x2c, blocks[1], growth_rate, name='block_2', 
                      data_format=data_format)   
    x2d = down_block(x2, comprs[1], name='down_2', 
                     data_format=data_format)   
    
    #-------------------------------------------- Dense Block 03
    x3i = layers.AveragePooling2D(2, padding='same', 
                                  data_format=data_format)(x2i)
    x3c = layers.Concatenate(axis=bn_axis)([x3i,x2d])   
    x3  = dense_block(x3c, blocks[2], growth_rate, name='block_3', 
                      data_format=data_format)    
    x3d = down_block(x3, comprs[2], name='down_3', 
                     data_format=data_format)   					  
    
    #-------------------------------------------- Dense Block 04
    x4i = layers.AveragePooling2D(2, padding='same', 
                                  data_format=data_format)(x3i)
    x4c = layers.Concatenate(axis=bn_axis)([x4i,x3d])
    x4  = dense_block(x4c, blocks[3], growth_rate, name='block_4', 
                      data_format=data_format)  
    up4 = upsampling_block(x4, comprs[3], name='up_4', 
                           data_format=data_format) 

    #-------------------------------------------- Dense Block 05
    x5  = layers.Concatenate(axis=bn_axis)([x3,up4])
    x5  = dense_block(x5, blocks[4], growth_rate, name='block_5', 
                      data_format=data_format)  
    up5 = upsampling_block(x5, comprs[4], name='up_5', 
                           data_format=data_format)   
      
    #-------------------------------------------- Dense Block 06
    x6  = layers.Concatenate(axis=bn_axis)([x2,up5])
    x6  = dense_block(x6, blocks[5], growth_rate, name='block_6', 
                      data_format=data_format)  
    up6 = upsampling_block(x6, comprs[5], name='up_6', 
                           data_format=data_format)  
    
    #-------------------------------------------- Dense Block 07
    x7 = layers.Concatenate(axis=bn_axis)([x1,up6])
    x7 = dense_block(x7, blocks[6], growth_rate, name='block_7', 
                     data_format=data_format)  
    x7 = layers.Conv2D(2, (1, 1), activation='elu', padding='same',
                       data_format=data_format)(x7)
   
    #-------------------------------------------- RESHAPE BLOCK  
    # This part is not yet prepared for 'channels_first'
    x8 = layers.Permute((3,1,2))(x7)  
    x8 = layers.Reshape((2, int(image_shape[0]*image_shape[1])))(x8)
    x8 = layers.Permute((2,1))(x8)
    x8 = layers.Activation('softmax')(x8)

    #-------------------------------------------- PARAMETERS
    model = Model(inputs=inputs, outputs=x8)
    nadam = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    if weights is not None:
        model.load_weights(weights)
    model.compile(optimizer=nadam, 
                  loss= 'categorical_crossentropy', 
                  metrics=['categorical_accuracy'])    
    return model

