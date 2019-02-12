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

#from ROOT import *
# from ROOT import TLorentzVector
# from lorentz import *

import tensorflow as tf
import numpy as np

#######################################


def entropy(model):
    H = 0.
    for layer in model.layers:
        conf = layer.get_config()

        if not "leaky_re_lu" in conf['name']:
            continue
        print conf

        weights = layer.get_weights()
        print weights
        shape = weights.shape
        print weights.shape

    return H


def filters_entropy(model):
    H = 0.
    for layer in model.layers:
        conf = layer.get_config()
        if not "conv2d" in conf['name']:
            continue
        print conf['name'], conf['filters']
        weights = layer.get_weights()[0]
        shape = weights.shape
        #print weights.shape
        kernel_size = shape[0]
        nchannels = shape[2]
        nfilters = shape[3]
        #print nchannels, nfilters, "(%i,%i)" % (kernel_size, kernel_size)
        for ichannel in range(nchannels):
            for ifilter in range(nfilters):
                w = weights[:, :, ichannel, ifilter]
                #print w
                H_i = sum(sum(w*np.log2(abs(w))))
                # print "%s :: ch=%i : filt=%i, H_i=%.3f" % (
                #    model.name, ichannel, ifilter, H_i)
                H += H_i
    return H

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def clip_weights(discriminator, c=0.01):
    weights = [np.clip(w, -c, c) for w in discriminator.get_weights()]
    discriminator.set_weights(weights)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_pairwise_distances(x,y):
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def g(x,y):
    sigmas = 0.1 * np.ones(7, dtype="float32")
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def mmd_loss(y_true, y_pred):
#    alpha = 1.0
#    loss  = alpha * tf.reduce_mean( g(y_true, y_true) ) \
#          - tf.reduce_mean( g(y_pred, y_pred) ) \
#          -(alpha - 1.0) * tf.reduce_mean( g(y_true, y_pred) )

    loss = tf.reduce_mean( g(y_true, y_true) ) \
         + tf.reduce_mean( g(y_pred, y_pred) ) \
         - 2.*tf.reduce_mean( g(y_true, y_pred) )
#    tf.where(loss > 0, loss, 0, name='value')

    return loss

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def PxPyPzE_to_PtEtaPhiM(x):
    # p = TLorentzVector()
    # p.SetPxPyPzE( x[0], x[1], x[2], x[3] )
    p = FourMomentum(x[0], x[1], x[2], x[3])

    return (p.Pt(), p.Eta(), p.Phi(), p.M())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def flip_eta(x):
    # return x

    # features = [
    #"ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
    #"ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
    #"jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M",
    #"jj_dPhi",  "jj_dEta",   "jj_dR",
    #]

    x_size = 18
    # x_size = 15
    # x_size = 10
    mask = np.ones(x_size, dtype="float32")
    mask[[1, 6, 11, 16]] = -1
    # mask[[1,6,11]] = -1
    # mask[[1,6]] = -1
    # mask = K.variable(value=mask, dtype='float64', name='mask')
    a = tf.identity(mask)

    y = a * x
    # y = tf.multiply( x, a )

    # K.print_tensor( x, "x =" )
    # K.print_tensor( a, "a =" )
    # K.print_tensor( y, "y =" )

    return y

#######################################


def make_generator_mlp_LorentzVector(GAN_noise_size):
    G_input = Input(shape=(GAN_noise_size,))

    G = Dense(GAN_noise_size, kernel_initializer='glorot_normal')(G_input)
    G = Activation('tanh')(G)
    # G = BatchNormalization()(G) #0.8

    G = Dense(64, kernel_initializer='glorot_normal')(G_input)
    G = Activation('tanh')(G)
    # G = BatchNormalization()(G) #0.8

    # G = Dense( 32 )(G)
    # G = Activation('tanh')(G)

    j1_PtEtaPhiEM = Dense(64, activation='tanh')(G)
    j1_PtEtaPhiEM = Dense(32, activation='tanh')(j1_PtEtaPhiEM)
    j1_PtEtaPhiEM = Dense(5)(j1_PtEtaPhiEM)

    j2_PtEtaPhiEM = Dense(64, activation='tanh')(G)
    j2_PtEtaPhiEM = Dense(32, activation='tanh')(j2_PtEtaPhiEM)
    j2_PtEtaPhiEM = Dense(5)(j2_PtEtaPhiEM)

    # j1_PxPyPzE = Dense(4, activation='tanh')(G)
    # j2_PxPyPzE = Dense(4, activation='tanh')(G)

    jj = concatenate([j1_PtEtaPhiEM, j2_PtEtaPhiEM])
    # jj = concatenate( [ j1_PxPyPzE, j2_PxPyPzE ] )
    # jj_PxPyPzE = add( [ j1_PxPyPzE, j2_PxPyPzE ] )

    jj_PtEtaPhiEM = Dense(64, activation='tanh')(jj)
    jj_PtEtaPhiEM = Dense(32, activation='tanh')(jj_PtEtaPhiEM)
    jj_PtEtaPhiEM = Dense(5)(jj_PtEtaPhiEM)

    jj_angles = Dense(32, activation='tanh')(jj)
    jj_angles = Dense(3)(jj_angles)

    G_output = concatenate([
        # j1_PxPyPzE,
        j1_PtEtaPhiEM,
        # j2_PxPyPzE,
        j2_PtEtaPhiEM,
        # jj_PxPyPzE,
        jj_PtEtaPhiEM,
        jj_angles
    ])
    generator = Model(G_input, G_output)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

    G = Dense(GAN_output_size, activation="tanh")(G)

    generator = Model(G_input, G)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~


def make_discriminator_mlp(GAN_output_size):
    # Build Discriminative model ...
    inshape = (GAN_output_size, )
    D_input = Input(shape=inshape, name='D_input')

    D = Dense(128)(D_input)
    D = LeakyReLU(0.2)(D)
    #D = Activation('tanh')(D)
    D = BatchNormalization(momentum=0.99)(D)  # 0.8

    D = Dense(64)(D)
    D = LeakyReLU(0.2)(D)
    #D = Activation('tanh')(D)
    D = BatchNormalization(momentum=0.99)(D)

    #D = Dense(32)(D)
    #D = LeakyReLU(0.2)(D)
    #D = Activation('tanh')(D)
    #D = BatchNormalization(momentum=0.99)(D)

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

    G_input = Input(shape=(GAN_noise_size,))

    reg = None  # l2(0.001)

    G = Dense(128, kernel_initializer='glorot_uniform')(G_input)
    #G = Dropout(0.2)(G)
    G = LeakyReLU(alpha=0.2)(G)
    #G = Activation("relu")(G)
    G = BatchNormalization()(G)

    G = Reshape([8, 8, 2])(G)  # default: channel last

    G = Conv2DTranspose(8, kernel_size=(2, 2), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    #G = Activation("relu")(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Conv2DTranspose(4, kernel_size=(3, 3), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    #G = Activation("relu")(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Conv2DTranspose(1, kernel_size=(4, 4), strides=1,
                        padding="same", kernel_regularizer=reg)(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)

    G = Flatten()(G)

    G_output = Dense(GAN_output_size)(G)
    G_output = Activation("tanh")(G_output)
    #G_output = Dense(GAN_output_size)(G)
    #G_output = LeakyReLU(0.2)(G_output)
    generator = Model(G_input, G_output)

    return generator

#~~~~~~~~~~~~~~~~~~~~~~


def make_discriminator_cnn(GAN_output_size):
    # Build Discriminative model ...
    # print "DEBUG: discriminator: input features:", GAN_output_size

    D_input = Input(shape=(GAN_output_size,))

    reg = None  # l2(0.001)

    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)

    D = Conv2D(64, kernel_size=(2, 2), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Conv2D(16, kernel_size=(4, 4), strides=1,
               padding="same", kernel_regularizer=reg)(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Flatten()(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)

    D = Dropout(0.2)(D)

    D_output = Dense(1, activation="sigmoid")(D)
#    D_output = Dense(1)(D)

    discriminator = Model(D_input, D_output)
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
