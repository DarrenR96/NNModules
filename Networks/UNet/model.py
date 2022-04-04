import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class UNetBlockDownSample(layers.Layer):
    def __init__(self, numFilters, size=4, strides=(2, 2), padding='same', **kwargs):
        super().__init__()
        self.numFilters = numFilters
        self.size = size
        self.strides = strides
        self.padding = padding
        
        self.conv1 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu1 = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu2 = layers.LeakyReLU()
        self.convP = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.reluP = layers.LeakyReLU()

        self.add = layers.Add()
        self.outputConv = layers.Conv2D(self.numFilters, self.size, (2,2), "same")
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
        x_0 = self.relu1(x_0,training=training)
        x_0 = self.conv2(x_0,training=training)
        x_0 = self.relu2(x_0, training=training)
        x_1 = self.convP(inputs,training=training)
        x_1 = self.reluP(x_1,training=training)
        x = self.add([x_0, x_1], training=training)
        x = self.outputConv(x,training=training)
        x = self.outputRelu(x,training=training)
        return x

class UNetBlockUpSample(layers.Layer):
    def __init__(self, numFilters, size=4, strides=(2, 2), padding='same', **kwargs):
        super().__init__()
        self.numFilters = numFilters
        self.size = size
        self.strides = strides
        self.padding = padding
        
        self.conv1 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu1 = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu2 = layers.LeakyReLU()
        self.convP = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.reluP = layers.LeakyReLU()

        self.add = layers.Add()
        self.outputConv = layers.Conv2DTranspose(self.numFilters, self.size, (2,2), "same")
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
        x_0 = self.relu1(x_0,training=training)
        x_0 = self.conv2(x_0,training=training)
        x_0 = self.relu2(x_0, training=training)
        x_1 = self.convP(inputs,training=training)
        x_1 = self.reluP(x_1, training=training)
        x = self.add([x_0, x_1], training=training)
        x = self.outputConv(x, training=training)
        x = self.outputRelu(x,training=training)
        return x

class UNet(keras.Model):
    def __init__(self, filters=[16,32,64,128,256,512], inputSize=192, outputSize=3):
        super().__init__()
        self.filters = filters
        self.network = []
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.residualConnection = layers.Add()
        for filter in self.filters:
            self.network.append(UNetBlockDownSample(filter))
        
        for filter in self.filters[::-1][1:]:
            self.network.append(UNetBlockUpSample(filter))

        self.finalConv2D = layers.Conv2DTranspose(3,4,(2,2),padding='same')

    def call(self, x, training=False):
        xIn = x[:,:,:,3:6]
        for filter in self.network:
            x = filter(x)
        x = self.finalConv2D(x,training=training)
        x = self.residualConnection([xIn, x])
        return x
    
    def model(self):
        x = keras.Input(shape=(self.inputSize,self.inputSize,9))
        return keras.Model(inputs=[x], outputs=self.call(x))
