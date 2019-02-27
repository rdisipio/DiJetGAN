import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, LSTM, Permute, Reshape, Masking, TimeDistributed, MaxPooling1D, Flatten, Bidirectional
from keras.layers.merge import *
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import concatenate, maximum, dot, average, add, subtract
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import Conv1D, GlobalMaxPooling1D, Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D
from keras.layers.merge import *
from keras.optimizers import *
from keras.regularizers import *
from keras.models import load_model

# from ROOT import *
# from ROOT import TLorentzVector
# from lorentz import *

import tensorflow as tf
import numpy as np

#######################################


def make_encoder(n_in, n_lat):
    input_E = Input((n_in,))
    encoded = Dense(128, activation='tanh')(input_E)
    encoded = Dense(64, activation='tanh')(encoded)
    encoded = Dense(32, activation='tanh')(encoded)
    output_E = Dense(n_lat, activation="tanh")(encoded)
    encoder = Model(input_E, output_E)

    return encoder

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def make_decoder(n_lat, n_out):
    input_D = Input((n_lat,))
    decoded = Dense(32, activation='tanh')(input_D)
    decoded = Dense(64, activation='tanh')(decoded)
    decoded = Dense(128, activation='tanh')(decoded)
    output_D = Dense(n_out, activation="tanh")(decoded)
    decoder = Model(input_D, output_D)

    return decoder

#################################


def clip_weights(discriminator, c=0.01):
    weights = [np.clip(w, -c, c) for w in discriminator.get_weights()]
    discriminator.set_weights(weights)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)


class GradNorm(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Be sure to call this somewhere!
        super(GradNorm, self).build(input_shape)

    def call(self, x):
        target, wrt = x
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_pairwise_distances(x, y):
    def norm(x): return tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def g(x, y):
    sigmas = 2. * np.ones(7, dtype="float32")
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def mmd_loss(y_true, y_pred):
    #    alpha = 1.0
    #    loss  = alpha * tf.reduce_mean( g(y_true, y_true) ) \
    #          - tf.reduce_mean( g(y_pred, y_pred) ) \
    #          -(alpha - 1.0) * tf.reduce_mean( g(y_true, y_pred) )

    loss = tf.reduce_mean(g(y_true, y_true)) \
        + tf.reduce_mean(g(y_pred, y_pred)) \
        - 2.*tf.reduce_mean(g(y_true, y_pred))
#    tf.where(loss > 0, loss, 0, name='value')

    return loss

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def gauss_loss(y_true, y_pred):
    beta = .5

#   rms = K.std(y_pred)
#   N   = tf.shape(y_pred)[0]
#   a   = np.power( 4./3., 1./5. )
#   p = tf.pow( N, -5 )
#   a = 1.059223841
#   p = 0.5
#   beta = a * rms / p

    y_diff = y_true-y_pred
    z = y_diff / beta
    # s = K.exp( -0.5*K.square(z) )
    s = 0.5*K.square(z)

    return K.mean(s)


#######################################


def make_generator_mlp(GAN_noise_size, GAN_output_size):
    # Build Generative model ...

    G_input = Input(shape=(GAN_noise_size,), name="Noise")

    G = Dense(128, kernel_initializer='glorot_normal')(G_input)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)  # 0.8

    G = Dense(64)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Dense(32)(G)
    G = LeakyReLU(alpha=0.2)(G)

    G_output = Dense(GAN_output_size, activation="tanh")(G)

    generator = Model(G_input, G_output)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~


def make_discriminator_mlp(GAN_output_size):
    # Build Discriminative model ...
    inshape = (GAN_output_size, )
    D_input = Input(shape=inshape, name='D_input')

    D = Dense(128, activation="tanh")(D_input)
#    D = LeakyReLU(0.2)(D)
    # D = Activation('tanh')(D)
    D = BatchNormalization(momentum=0.99)(D)  # 0.8

    D = Dense(64, activation="tanh")(D)
    #D = LeakyReLU(0.2)(D)
    # D = Activation('tanh')(D)
    D = BatchNormalization(momentum=0.99)(D)

    # D = Dense(32)(D)
    # D = LeakyReLU(0.2)(D)
    # D = Activation('tanh')(D)
    # D = BatchNormalization(momentum=0.99)(D)

    # D = Dense( 8 )(D)
    # D = Activation('tanh')(D)
    # D = BatchNormalization(momentum=0.99)(D)

    # D = Dense( 4 )(D)
    # D = Activation('elu')(D)

    # D_output = Dense( 2, activation="softmax")(D)
    D_output = Dense(1, activation="sigmoid")(D)
    discriminator = Model(D_input, D_output)
    # discriminator.compile( loss='categorical_crossentropy', optimizer=dopt )

    return discriminator


##########################


def make_generator_cnn(GAN_noise_size, GAN_output_size):
    # Build Generative model ...

    G_input = Input(shape=(GAN_noise_size,), name="G_in")

    reg = None  # l2(0.001)

    G = Dense(128, kernel_initializer='glorot_uniform')(G_input)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Reshape([8, 8, 2])(G)  # default: channel last

    G = Conv2DTranspose(32, kernel_size=(2, 2), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Conv2DTranspose(16, kernel_size=(3, 3), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Flatten()(G)

    G_output = Dense(GAN_output_size)(G)
    G_output = Activation("tanh")(G_output)
    generator = Model(G_input, G_output)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~


def make_generator_cnn_cgan(GAN_noise_size, GAN_output_size):

    G_input_noise = Input(shape=(GAN_noise_size,), name="G_in_noise")
    G_input_dijet = Input((1,), name="G_in_jj")

    G_input = concatenate([G_input_noise, G_input_dijet])

    reg = None  # l2(0.001)

    G = Dense(128, kernel_initializer='glorot_uniform')(G_input)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Reshape([8, 8, 2])(G)  # default: channel last

    G = Conv2DTranspose(32, kernel_size=(2, 2), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Conv2DTranspose(16, kernel_size=(3, 3), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Flatten()(G)

    G_output = Dense(GAN_output_size)(G)
    G_output = Activation("tanh")(G_output)

    generator = Model([G_input_noise, G_input_dijet], G_output)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~


def make_discriminator_cnn(GAN_output_size):
    D_input = Input(shape=(GAN_output_size,), name="D_in")

    reg = None  # l2(0.001)

    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)

    D = Conv2D(64, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    # D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(16, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    # D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Flatten()(D)
    # D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D_output = Dropout(0.2)(D)
    D_output = Dense(1, activation="sigmoid")(D_output)

    discriminator = Model(D_input, D_output)
    return discriminator

#~~~~~~~~~~~~~~~~~~~~~~


def make_discriminator_cnn_cgan(GAN_output_size):
    D_input_p4 = Input(shape=(GAN_output_size,), name="D_in_p4")
    D_input_jj = Input((1,), name="D_in_jj")

    D_input = concatenate([D_input_p4, D_input_jj])

    reg = None  # l2(0.001)

    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)

    D = Conv2D(64, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    # D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(16, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    # D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Flatten()(D)
    # D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D_output = Dropout(0.2)(D)
    D_output = Dense(1, activation="sigmoid")(D_output)

    discriminator = Model([D_input_p4, D_input_jj], D_output)
    return discriminator

##########################


def make_generator_rnn(GAN_noise_size, GAN_output_size):

    G_input = Input(shape=(GAN_noise_size,))

    # reg.regularizers.l2(0.001)

    G = Dense(128, kernel_initializer='glorot_normal')(G_input)
    G = Activation('tanh')(G)
    G = BatchNormalization(momentum=0.99)(G)  # 0.8

    G = Reshape((32, 4))(G)

    # G = Bidirectional( LSTM( 32, return_sequences=True  ) )(G)
    # G = Bidirectional( LSTM( 8, return_sequences=True ) )(G)
    G = LSTM(32, return_sequences=True)(G)
    # kernel_regularizer=regularizers.l2(0.01)
    G = LSTM(16, return_sequences=False)(G)
    G = Activation('tanh')(G)

    G = Dense(GAN_output_size, activation="tanh")(G)

    generator = Model(G_input, G)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~


def make_discriminator_rnn(GAN_output_size):

    inshape = (n_features, )
    D_input = Input(shape=inshape, name='D_input')

    D = Dense(128, kernel_initializer='glorot_normal')(D_input)
    D = Activation('tanh')(D)
    D = Reshape((16, 8))(D)

    # D = Bidirectional( LSTM( 16, return_sequences=True  ) )(D)

    D = Bidirectional(LSTM(8, return_sequences=False))(D)
    D = Activation('tanh')(D)

    # D_output = Dense( 2, activation="softmax")(D)
    D_output = Dense(1, activation="sigmoid")(D)
    discriminator = Model(D_input, D_output)
    # discriminator.compile( loss='categorical_crossentropy', optimizer=dopt )

    return discriminator


##########################
