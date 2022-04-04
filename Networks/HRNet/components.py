## Custom Layers that create HrNet

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa

class hrLayers(layers.Layer):
    def __init__(self, numFilters, size, strides=(1, 1), padding='same', **kwargs):
        super().__init__()
        self.numFilters = numFilters
        self.size = size
        self.strides = strides
        self.padding = padding
        self.conv1 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu1_2 = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu1 = layers.LeakyReLU()
        self.conv3 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu3_4 = layers.LeakyReLU()
        self.conv4 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu2 = layers.LeakyReLU()
        self.convP = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.groupNorm = tfa.layers.GroupNormalization(4)
        self.add = layers.Add()
        self.outputRelu = layers.LeakyReLU()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'strides' : self.strides,
            'padding' : self.padding
        })
        return config
    
    def call(self, inputs, training=False):
        x_0 = self.conv1(inputs,training=training)
        x_0 = self.relu1_2(x_0, training=training)
        x_0 = self.conv2(x_0,training=training)
        x_0 = self.relu1(x_0,training=training)
        x_0 = self.conv3(x_0,training=training)
        x_0 = self.relu3_4(x_0,training=training)
        x_0 = self.conv4(x_0,training=training)
        x_0 = self.relu2(x_0, training=training)
        x_1 = self.convP(inputs,training=training)
        x_1 = self.groupNorm(x_1,training=training)
        x = self.add([x_0, x_1], training=training)
        x = self.outputRelu(x,training=training)
        return x


class hrBlock(layers.Layer):
    def __init__(self, numFilters, kernelSize, inputSize, numLayersPerBlock=6):
        super().__init__()
        self.numFilters = numFilters
        self.inputSize = inputSize
        self.numLayersPerBlock = numLayersPerBlock
        self.kernelSize = kernelSize
        self.block = []
        for _ in range(self.numLayersPerBlock):
            self.block.append(hrLayers(self.numFilters, self.kernelSize))
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'inputSize' : self.inputSize,
            'kernelSize': self.kernelSize
        })
        return config
    
    def call(self, inputs, training=False):
        # Tensor shape is (batchSize, height, width, channels)
        for i in range(len(inputs)):
            # Check width/height
            _shape = inputs[i].shape
            _shape = (_shape[1],_shape[2])
            if _shape[0] != _shape[1]:
                raise TypeError(f"Internal height({_shape[0]}) and width({_shape[1]}) are not equal")    
            # Downsample
            if _shape[0] > self.inputSize:
                _ratio = _shape[0]//self.inputSize
                inputs[i] = layers.AveragePooling2D(_ratio)(inputs[i])
            # Upsample 
            if _shape[0] < self.inputSize:
                _ratio = self.inputSize//_shape[0]
                inputs[i] = layers.UpSampling2D(_ratio)(inputs[i])
        
        x_0 = tf.concat(inputs,-1)
        for i in range(len(self.block)):
            x_0 = self.block[i](x_0, training=training)
        return x_0
    

class hrFinalBlock(layers.Layer):
    def __init__(self, filters=[512,128,64,32,9,3], kernelSize=5):
        super().__init__()
        self.filters = filters
        self.Conv2D = []
        self.kernelSize = kernelSize

        for filter in self.filters:
            self.Conv2D.append(layers.Conv2D(filter, self.kernelSize, padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform()))
            self.Conv2D.append(layers.LeakyReLU())

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernelSize' : self.kernelSize
        })
        return config

    def call(self, x, training=False):
        for filter in self.Conv2D:
            x = filter(x, training=training)
        return x
