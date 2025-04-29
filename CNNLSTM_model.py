import tensorflow as tf
from keras.optimizers import Adam,SGD, RMSprop, Adagrad, nadam,Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from keras_layer_normalization import LayerNormalization
from keras.layers import Input, Convolution3D, MaxPooling3D, BatchNormalization, Flatten, Dense, Dropout,AveragePooling3D, Activation
from keras.layers import LSTM, Conv2D, MaxPool2D,TimeDistributed,CuDNNLSTM,MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout,Reshape
from keras.layers import Flatten
from keras.layers import TimeDistributed

# Build the model
model = Sequential()
model.add(Conv2D(filters=64,
                 kernel_size=(11, 3),
                 strides=(2, 1),
                 kernel_initializer='he_normal', kernel_regularizer=l2(0),
                 padding='same', input_shape=(512, 6, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 1)))
for m in range(0, 3):
    model.add(Conv2D(filters=32 * (1 + 1),
                     kernel_size=(7, 6),
                     kernel_initializer='he_normal', kernel_regularizer=l2(0), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 1)));
model.add(TimeDistributed(Dense(128, name="first_dense" )))
model.add(Flatten())
model.add(Reshape((8064,4), input_shape=(32256,1)))
model.add(LSTM(128, kernel_regularizer=l2(0), kernel_initializer='he_normal',input_shape=(8064,4)))
model.add(LayerNormalization(trainable=True))
model.add(Activation("relu"))
model.add(Reshape((16,8), input_shape=(128, 1)))
model.add(LSTM(32, kernel_regularizer=l2(0), kernel_initializer='he_normal',input_shape=(16,8)))
model.add(LayerNormalization(trainable=True))
model.add(Activation("relu"))
# Final fully Connected layer specifying outputs
model.add(Dense(6, activation="linear", name="FC_out"))

# Compile model, produce summary and save model image to file
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

