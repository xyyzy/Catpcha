# %%

import os
import keras as K
from keras.metrics import categorical_accuracy

from DataGenerator import DataGenerator
from keras import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Concatenate, Reshape, LeakyReLU
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.utils import plot_model

# %%

input_shape = (128, 128, 3)

# model_data
K.backend.clear_session()

X_input = Input(shape=input_shape)
X = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(X_input)
X = BatchNormalization()(X)
X = LeakyReLU()(X)
X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

X = Flatten()(X)
X = Dropout(rate=0.5)(X)
X1 = Dense(62, activation='softmax')(X)
X2 = Dense(62, activation='softmax')(X)
X3 = Dense(62, activation='softmax')(X)
X4 = Dense(62, activation='softmax')(X)
X = Concatenate(axis=-1)([X1, X2, X3, X4])
predict = Reshape(target_shape=(4, 62))(X)

# %%

model = Model(inputs=X_input, outputs=predict, name='net')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', categorical_accuracy])
plot_model(model=model, show_layer_names=True, show_shapes=True)

# model_data.summary()
# %%

basepath = r'F:\Code\DL\CAPTCHA'
data_file = os.path.join(basepath, 'data', 'train', 'train_label.csv')
data_dir = os.path.join(basepath, 'data', 'train')
batch_size = 32
data_gen = DataGenerator(data_file=data_file, data_dir=data_dir, batch_size=batch_size, shuffle=True)

# %%
model.fit_generator(generator=data_gen, epochs=100, workers=4)
