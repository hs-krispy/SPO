import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, Conv2DTranspose, GaussianNoise
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class FeatureExtractor:
    def __init__(self, dims, feature_shape):
        self.dims = dims
        self.feature_shape = feature_shape
        self.n_stacks = len(dims)

    def cae(self, act="relu", initializer="he_normal", noise=False):

        input = Input(shape=self.dims[0], name='input')
        h = input
        if noise:
            h = GaussianNoise(0.2)(h)
        for i in range(self.n_stacks - 1):
            if i % 2 == 0:
                h = Conv2D(self.dims[i + 1], kernel_size=3, strides=2, padding="same", activation=act,
                           kernel_initializer=initializer,
                           name=f'encoder_{i + 1}')(h)
            else:
                h = Conv2D(self.dims[i + 1], kernel_size=3, strides=1, padding="same", activation=act,
                           kernel_initializer=initializer,
                           name=f'encoder_{i + 1}')(h)
        reshape = h.shape[1:]
        h = Flatten()(h)
        shape = h.shape[1]
        encoder_output = Dense(self.feature_shape, name="Feature")(h)

        h = Dense(shape, activation=act)(encoder_output)
        h = tf.keras.layers.Reshape(reshape, input_shape=(shape,))(h)
        for i in range(self.n_stacks - 1, 0, -1):
            if i % 2 == 1:
                h = Conv2DTranspose(self.dims[i], kernel_size=3, strides=2, padding="same", activation=act,
                                    kernel_initializer=initializer, name=f'decoder_{i}')(h)
            else:
                h = Conv2DTranspose(self.dims[i], kernel_size=3, strides=1, padding="same", activation=act,
                                    kernel_initializer=initializer, name=f'decoder_{i}')(h)
        reconstructed_img = Conv2DTranspose(self.dims[0][2], kernel_size=3, strides=1, padding="same", activation="sigmoid", name='output')(
            h)

        return Model(inputs=input, outputs=reconstructed_img)


x = np.load("./Data/ions_internal_scaling_dataset_107(800x400).npy")
x = x / 255.

batch_size = 16
epoch = 100
optimizer = Adam()
# optimizer = RMSprop()
# loss = "mse"
loss = "binary_crossentropy"

def scheduler(epoch, lr):
    if epoch > 0:
        lr *= 0.99

    return lr

dimensions = [x.shape[1:], 32, 64, 128, 64, 32]
CAE = FeatureExtractor(dims=dimensions, feature_shape=512)
CAE = CAE.cae()
CAE.summary()

pretrain_time = time()
# Train autoencoder for feature extract
callbacks = [EarlyStopping(monitor="loss", patience=5, verbose=1, restore_best_weights=True),
             LearningRateScheduler(scheduler)]
CAE.compile(loss=loss, optimizer=optimizer)
# plot_model(CAE, to_file="CAE.png", show_shapes=True)
history = CAE.fit(x, x, batch_size=batch_size, epochs=epoch, callbacks=callbacks, verbose=1)
pretrain_time = time() - pretrain_time
# autoencoder.save_weights(f'weights/{epoch}epoch-ae_weights.h5')
print(f"Time to train the autoencoder: {str(pretrain_time / 60)}m")

method = "Ions"
feature = CAE.get_layer("Feature").output
encoder = Model(inputs=CAE.input, outputs=feature)
encoder.summary()
# model save
encoder.save(f"./models/{method}_{epoch}_{batch_size}_{loss}_encoder.h5")
