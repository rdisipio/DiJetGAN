#!/usr/bin/env python

import os
import sys

try:
    import cPickle as pickle
except:
    import pickle

from ROOT import *
rng = TRandom3()

import numpy as np

from sklearn.preprocessing import MinMaxScaler

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
from keras.callbacks import *
from keras.utils import plot_model

import helper_functions as hf

def GenerateTrainingSample(X_train, y_train):
    print "INFO: generating %i four-momenta" % n_train

    p = TLorentzVector()

    for i in range(n_train):

        pt  = rng.Uniform(0., 1500.)
        eta = rng.Uniform(-2.5, 2.5)
        phi = rng.Uniform(-np.pi, np.pi)
        m   = rng.Uniform(0., 500.)
        p.SetPtEtaPhiM(pt, eta, phi, m)

        X_train[i][0] = p.Pt()
        X_train[i][1] = p.Eta()
        X_train[i][2] = p.Phi()
        X_train[i][3] = p.M()

        y_train[i][0] = p.Px()
        y_train[i][1] = p.Py()
        y_train[i][2] = p.Pz()
        y_train[i][3] = p.E()

#~~~~~~~~~~~~~~~~~~~~

optimizer = Adam(0.001)
#optimizer = SGD(0.001)

input_PtEtaPhiM = Input( (4,) )
encoded = Dense(4)(input_PtEtaPhiM)
encoded = LeakyReLU(0.2)(encoded)
encoded = Dense(64)(encoded)
encoded = LeakyReLU(0.2)(encoded)
encoded = Dense(256)(encoded)
encoded = LeakyReLU(0.2)(encoded)
encoded = Dense(4)(encoded)
output_PxPyPzE = LeakyReLU(0.2)(encoded)

decoded = Dense(4)(output_PxPyPzE)
decoded = LeakyReLU(0.2)(decoded)
decoded = Dense(64)(decoded)
decoded = LeakyReLU(0.2)(decoded)
decoded = Dense(256)(decoded)
decoded = LeakyReLU(0.2)(decoded)
decoded = Dense(4)(decoded)
output_PtEtaPhiM = LeakyReLU(0.2)(decoded)

autoencoder = Model( input_PtEtaPhiM, output_PtEtaPhiM )

# PtEtaPhiM -> PxPyPzE
encoder     = Model( input_PtEtaPhiM, output_PxPyPzE )
encoder.name = "PtEtaPhiM_to_PxPyPzE"

# PxPyPzE -> PtEtaPhiM
input_PxPyPzE = Input( (4,) )
decoder_layer = autoencoder.layers[-1]
decoder = Model( input_PxPyPzE, decoder_layer(input_PxPyPzE) )
decoder.name = "PxPyPzE_to_PtEtaPhiM"

autoencoder.name = "autoencoder"
autoencoder.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['accuracy'])
autoencoder.summary()

plot_model( autoencoder, show_shapes=True, to_file="img/model_autoencoder.png")

# n_train = 10
n_train = 1000000

X_train = np.zeros([n_train, 4])
y_train = np.zeros([n_train, 4])

GenerateTrainingSample( X_train, y_train )

print "INFO: done generating."

print "INFO: (pT,eta,phi,M) before transformation:"
print y_train

scaler_PtEtaPhiM = hf.FourMomentumScaler( "PtEtaPhiM" )
scaler_PtEtaPhiM.transform(X_train)

scaler_PxPyPzE = hf.FourMomentumScaler( "PxPyPzE" )
scaler_PxPyPzE.transform(y_train)

print "INFO: (px,py,pz,E) after transformation:"
print y_train

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
    # ModelCheckpoint(filepath=weights_filename, monitor='val_loss', save_best_only=True)
]

N_EPOCHS = 30
BATCH_SIZE = 1024

history = autoencoder.fit(X_train, y_train,
                  epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.20, shuffle=True,
                  callbacks=callbacks)
score = autoencoder.evaluate(X_train, y_train)
print
print "Score:", score

y_PxPyPzE   = encoder.predict(X_train)
y_PtEtaPhiM = decoder.predict(y_PxPyPzE)

scaler_PxPyPzE.inverse_transform( y_PxPyPzE )
scaler_PtEtaPhiM.inverse_transform( y_PtEtaPhiM )

PtEtaPhiM_to_PxPyPzE_filename = "lorentz/PtEtaPhiM_to_PxPyPzE.h5"
PxPyPzE_to_PtEtaPhiM_filename = "lorentz/PxPyPzE_to_PtEtaPhiM.h5"

encoder.save( PtEtaPhiM_to_PxPyPzE_filename )
decoder.save( PxPyPzE_to_PtEtaPhiM_filename )

print "INFO: PtEtaPhiM_to_PxPyPzE model saved to file", PtEtaPhiM_to_PxPyPzE_filename
print "INFO: PxPyPzE_to_PtEtaPhiM model saved to file", PxPyPzE_to_PtEtaPhiM_filename

for i in range(10):
    x_PtEtaPhiM = TLorentzVector()
    x_PtEtaPhiM.SetPtEtaPhiM( X_train[i][0], X_train[i][1], X_train[i][2], X_train[i][3] )

    v_PxPyPzE = TLorentzVector()
    v_PxPyPzE.SetPxPyPzE( y_PxPyPzE[i][0], y_PxPyPzE[i][1], y_PxPyPzE[i][2], y_PxPyPzE[i][3] )

    p_PtEtaPhiM = TLorentzVector()
    p_PtEtaPhiM.SetPtEtaPhiM( y_PtEtaPhiM[i][0], y_PtEtaPhiM[i][1], y_PtEtaPhiM[i][2], y_PtEtaPhiM[i][3] )

    print "%-5i )" % i
    print "(root) (pT,eta,phi,M) = (%4.1f,%3.2f,%3.2f,%4.1f)" % ( x_PtEtaPhiM.Pt(), x_PtEtaPhiM.Eta(), x_PtEtaPhiM.Phi(), x_PtEtaPhiM.M() )
    print "(root) (px,py,pz,E)   = (%4.1f,%4.1f,%4.1f,%4.1f)" % ( x_PtEtaPhiM.Px(), x_PtEtaPhiM.Py(), x_PtEtaPhiM.Pz(), x_PtEtaPhiM.E() )
    print "(encd) (px,py,pz,E)   = (%4.1f,%4.1f,%4.1f,%4.1f)" % ( v_PxPyPzE.Px(),   v_PxPyPzE.Py(),   v_PxPyPzE.Pz(),   v_PxPyPzE.E() )
    print "(decd) (pT,eta,phi,M) = (%4.1f,%3.2f,%3.2f,%4.1f)" % ( p_PtEtaPhiM.Px(), p_PtEtaPhiM.Py(), p_PtEtaPhiM.Pz(), p_PtEtaPhiM.M() )
    print "----------------"

