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

    v1 = TLorentzVector()
    v2 = TLorentzVector()
    p = TLorentzVector()

    for i in range(n_train):

        pt = rng.Uniform(0., 1500.)
        eta = rng.Uniform(-2.5, 2.5)
        phi = rng.Uniform(-np.pi, np.pi)
        m = rng.Uniform(0., 500.)
        v1.SetPtEtaPhiM(pt, eta, phi, m)

        X_train[i][0] = v1.Pt()
        X_train[i][1] = v1.Eta()
        X_train[i][2] = v1.Phi()
        X_train[i][3] = v1.M()

        pt = rng.Uniform(0., 1500.)
        eta = rng.Uniform(-2.5, 2.5)
        phi = rng.Uniform(-np.pi, np.pi)
        m = rng.Uniform(0., 500.)
        v2.SetPtEtaPhiM(pt, eta, phi, m)

        X_train[i][4] = v2.Pt()
        X_train[i][5] = v2.Eta()
        X_train[i][6] = v2.Phi()
        X_train[i][7] = v2.M()

        p = v1 + v2

        y_train[i][0] = p.Pt()
        y_train[i][1] = p.Eta()
        y_train[i][2] = p.Phi()
        y_train[i][3] = p.M()


#~~~~~~~~~~~~~~~~~~~~


def make_model_cnn():
    input = Input(shape=(8,))

    x = Reshape((2, 4, 1))(input)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    # x = UpSampling2D(size=2)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=16, kernel_size=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)

    output = Dense(4)(x)

    model = Model(input, output)
    return model

#~~~~~~~~~~~~~~~~~~~~


def make_model_mlp_towers():

    input = Input(shape=(8,))

    x = Dense(8)(input)
    # x = Dense(256)(input)
    # x = LeakyReLU(alpha=0.2)(x)

    x_pt = Dense(128)(x)
    x_pt = LeakyReLU(alpha=0.2)(x_pt)

    x_eta = Dense(128)(x)
    x_eta = LeakyReLU(alpha=0.2)(x_eta)

    x_phi = Dense(128)(x)
    x_phi = LeakyReLU(alpha=0.2)(x_phi)

    x_m = Dense(128)(x)
    x_m = LeakyReLU(alpha=0.2)(x_m)

    x_out = concatenate([x_pt, x_eta, x_phi, x_m])
    x_out = Dense(32)(x_out)
    x_out = LeakyReLU(0.2)(x_out)

    output = Dense(4)(x_out)
    output = LeakyReLU(0.2)(output)

    model = Model(input, output)
    return model


#~~~~~~~~~~~~~~~~~~~~

def make_model_mlp():
    input = Input(shape=(8,))

    x = Dense(32)(input)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)

#    x = Dense(512)(x)
#    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(4)(x)
    output = LeakyReLU(0.2)(x)

    model = Model(input, output)
    return model

#~~~~~~~~~~~~~~~~~~~~


def make_model_rnn():
    input = Input(shape=(8,))

    x = Dense(8)(input)
    x = LeakyReLU(0.2)(x)

    x = Reshape((2, 4))(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(4)(x)
    x = LeakyReLU(0.2)(x)

    model = Model(input, x)
    return model

#~~~~~~~~~~~~~~~~~~~~

def make_model_sum():
    input = Input(shape=(8,))

    x_in   = Input((4,))
    x_conv = Dense(64)(x_in)
    x_conv = LeakyReLU(0.2)(x_conv)
    x_conv = Dense(128)(x_in)
    x_conv = LeakyReLU(0.2)(x_conv)
    x_conv = Dense(4)(x_in)
    x_out  = LeakyReLU(0.2)(x_conv)
    conv   = Model( x_in, x_out )

    p1 = Dense(4)(input)
    p1 = conv(p1)

    p2 = Dense(4)(input)
    p2 = conv(p2)

    pp = add( [p1, p2] )
    pp = Dense(64)(pp)
    pp = LeakyReLU(0.2)(pp)
    pp = Dense(128)(pp)
    pp = LeakyReLU(0.2)(pp)
    pp = Dense(4)(pp)
    pp = LeakyReLU(0.2)(pp)

    model = Model(input, pp)
    return model

def make_model():
    #return make_model_cnn()
    #return make_model_mlp()
    #return make_model_mlp_towers()
    # return make_model_rnn()
    return make_model_sum()

#~~~~~~~~~~~~~~~~~~~~


optimizer = Adam(0.001)
#optimizer = SGD(0.001)

dnn = make_model()
dnn.name = "SumP4"
dnn.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['accuracy'])
dnn.summary()

plot_model(dnn, show_shapes=True, to_file="img/model_sumP4.png")

# n_train = 10
n_train = 1000000

X_train = np.zeros([n_train, 8])
y_train = np.zeros([n_train, 4])

GenerateTrainingSample(X_train, y_train)

print "INFO: done generating."

print "INFO: (pT,eta,phi,M) before transformation:"
print y_train

# X_scaler = MinMaxScaler((-1,1))
# y_scaler = MinMaxScaler((-1,1))

# X_train = X_scaler.fit_transform( X_train )
# y_train = y_scaler.fit_transform( y_train )

P4_scaler = hf.FourMomentumScaler()
P4_scaler.transform(X_train)
P4_scaler.transform(y_train)

print "INFO: (pT,eta,phi,M) after transformation:"
print y_train

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),


    # ModelCheckpoint(filepath=weights_filename, monitor='val_loss', save_best_only=True)
]

N_EPOCHS = 30
BATCH_SIZE = 1024

history = dnn.fit(X_train, y_train,
                  epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.20, shuffle=True,
                  callbacks=callbacks)
score = dnn.evaluate(X_train, y_train)
print
print "Score:", score

y_hat = dnn.predict(X_train)
# y_hat = y_scaler.inverse_transform( y_hat )
# X_train = X_scaler.inverse_transform( X_train )
# y_train = y_scaler.inverse_transform( y_train )
P4_scaler.inverse_transform(y_hat)
P4_scaler.inverse_transform(X_train[:4])
P4_scaler.inverse_transform(X_train[4:])
P4_scaler.inverse_transform(y_train)

for i in range(10):
    p1 = TLorentzVector()
    p2 = TLorentzVector()

    p1.SetPtEtaPhiM(y_train[i][0], y_train[i][1], y_train[i][2], y_train[i][3])
    p2.SetPtEtaPhiM(y_hat[i][0], y_hat[i][1], y_hat[i][2], y_hat[i][3])

    print "%-5i ) (pT,eta,phi,E:M) = (%4.1f,%3.2f,%3.2f,%4.1f:%4.1f) :: (%4.1f,%3.2f,%3.2f,%4.1f:%4.1f)" % (
        i,
        p1.Pt(), p1.Eta(), p1.Phi(), p1.E(), p1.M(),
        p2.Pt(), p2.Eta(), p2.Phi(), p2.E(), p2.M())

model_filename = "lorentz/sumP4_PtEtaPhiM.h5"
dnn.save(model_filename)
print "INFO: model saved to file", model_filename
#weights_filename = "lorentz/sumP4_PtEtaPhiM.weights.h5"
#dnn.save_weights(weights_filename)
#print "INFO: weights saved to file", weights_filename

# scaler_filename = "scaler_sumP4_PtEtaPhiM.pkl"
# with open( scaler_filename, "wb" ) as file_scaler:
#   pickle.dump( y_scaler, file_scaler )
# print "INFO: scaler saved to file", scaler_filename

from ROOT import *
training_filename = "lorentz/training_history.sumP4_PtEtaPhiM.root"
training_file = TFile.Open(training_filename, "RECREATE")
n_epochs = len(history.history['acc'])
h_acc = TH1F("acc",      "Epoch;Training Accuracy",
             n_epochs, 0.5, n_epochs+0.5)
h_loss = TH1F("loss",     "Epoch;Training Loss",
              n_epochs, 0.5, n_epochs+0.5)
h_val_acc = TH1F("val_acc",  "Epoch;Validation Accuracy",
                 n_epochs, 0.5, n_epochs+0.5)
h_val_loss = TH1F("val_loss", "Epoch;Validation Loss",
                  n_epochs, 0.5, n_epochs+0.5)
for i in range(n_epochs):
    acc = float("%.3f" % history.history['acc'][i])
    val_acc = float("%.3f" % history.history['val_acc'][i])
    loss = float("%.3f" % history.history['loss'][i])
    val_loss = float("%.3f" % history.history['val_loss'][i])

    h_acc.SetBinContent(i+1, acc)
    h_loss.SetBinContent(i+1, loss)
    h_val_acc.SetBinContent(i+1, val_acc)
    h_val_loss.SetBinContent(i+1, val_loss)

training_file.Write()
training_file.Close()
print "INFO: training history saved to file", training_filename

print "INFO: done."
