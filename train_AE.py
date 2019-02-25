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
np.set_printoptions(precision=2, suppress=True, linewidth=300)

import pandas as pd

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


def GenerateTrainingSample(X_train):
    print "INFO: generating %i four-momenta" % n_train

    j1 = TLorentzVector()
    j2 = TLorentzVector()

    for i in range(n_train):

        # generate two random four momenta

        pt = rng.Uniform(0., 1500.)
        eta = rng.Uniform(-2.5, 2.5)
        phi = rng.Uniform(-np.pi, np.pi)
        m = rng.Uniform(0., 500.)
        j1.SetPtEtaPhiM(pt, eta, phi, m)

        pt = rng.Uniform(0., 1500.)
        eta = rng.Uniform(-2.5, 2.5)
        phi = rng.Uniform(-np.pi, np.pi)
        m = rng.Uniform(0., 500.)
        j2.SetPtEtaPhiM(pt, eta, phi, m)

        jj = j1 + j2
        jj.dEta = j1.Eta() - j2.Eta()
        jj.dPhi = j1.DeltaPhi(j2)
        jj.dR = j1.DeltaR(j2)

        X_train[i][0] = j1.Pt()
        X_train[i][1] = j1.Eta()
        X_train[i][2] = j1.Phi()
        X_train[i][3] = j1.E()
        X_train[i][4] = j1.M()

        X_train[i][5] = j2.Pt()
        X_train[i][6] = j2.Eta()
        X_train[i][7] = j2.Phi()
        X_train[i][8] = j2.E()
        X_train[i][9] = j2.M()

        X_train[i][10] = jj.Pt()
        X_train[i][11] = jj.Eta()
        X_train[i][12] = jj.Phi()
        X_train[i][13] = jj.E()
        X_train[i][14] = jj.M()

       # X_train[i][15] = jj.dEta
       # X_train[i][16] = jj.dPhi
       # X_train[i][17] = jj.dR

#~~~~~~~~~~~~~~~~~~~~


level = "ptcl"
if len(sys.argv) > 1:
    level = sys.argv[1]
training_filename = "csv/mg5_dijet_ht500.%s.pt250.nominal.csv" % (level)

from features import *
if level == "ptcl":
    features = [
        "ljet1_pt", "ljet1_eta", "ljet1_M",
        "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M",
        "jj_pt",    "jj_eta",    "jj_phi",    "jj_M",
        "jj_dPhi",  "jj_dEta",  "jj_dR",
    ]

else:
    features = [
        "ljet1_pt", "ljet1_eta", "ljet1_M",
        "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M",
        "jj_pt",    "jj_eta",    "jj_phi",    "jj_M",
        "jj_dPhi",  "jj_dEta",  "jj_dR",
        "mu",
    ]
data = pd.read_csv(training_filename, delimiter=',', names=header)
print "INFO: dataset loaded into memory"
print "INFO: header:"
print header

# print data.isnull().values.any()
print "INFO: checking if input data has NaNs"
nan_rows = data[data.isnull().T.any().T]
print nan_rows
data.dropna(inplace=True)
print "INFO: number of good events:", len(data)

X_train = data[features].values
print "INFO: X_train shape:", X_train.shape


# n_features = 3 * 5 + 3  # = 18, in case you were wondering
#n_features = 3*5
n_features = len(features)
n_latent = 8
compression_factor = float(n_features) / float(n_latent)
print "INFO: compression factor: %.3f" % compression_factor

# n_train = 10
# n_train = 1000000
# X_train = np.zeros([n_train, n_features])
# GenerateTrainingSample(X_train)
# print "INFO: done generating."

print "INFO: training sample before scaling:"
print X_train

if not os.path.exists("lorentz/"):
    print "INFO: creating output directory lorentz/"
    os.makedirs("lorent/")


scaler = MinMaxScaler([-1, 1])
X_train = scaler.fit_transform(X_train)
scaler_filename = "lorentz/scaler.%s.pkl" % (level)
with open(scaler_filename, "wb") as file_scaler:
    pickle.dump(scaler, file_scaler)
print "INFO: scaler saved to file", scaler_filename

print "INFO: training sample after transformation:"
print X_train


##################
# Define models

from models import *

encoder = make_encoder( n_features, n_latent )
encoder.name = "Encoder"
print "INOF: Encoder:"
encoder.summary()

decoder = make_decoder( n_latent, n_features )
decoder.name = "Decoder"
print "INFO: Decoder:"
decoder.summary()

input_AE = Input((n_features,))
latent_AE = encoder(input_AE)
output_AE = decoder(latent_AE)
autoencoder = Model(input_AE, output_AE)
autoencoder.name = "Autoencoder"
print "INFO: Autoencoder:"
autoencoder.summary()

optimizer = Adam(0.001)
# optimizer = SGD(0.001)
#optimizer = RMSprop(0.001)

autoencoder.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['accuracy'])

plot_model(autoencoder, show_shapes=True, to_file="img/model_autoencoder.png")

##################
# Training

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
    # ModelCheckpoint(filepath=weights_filename, monitor='val_loss', save_best_only=True)
]

N_EPOCHS = 10
BATCH_SIZE = 1024

history = autoencoder.fit(X_train, X_train,
                          epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                          validation_split=0.20,
                          shuffle=True,
                          verbose=1,
                          callbacks=callbacks)
print
print "INFO: training finished"
print

print "INFO: encoding input..."
y_encoded = encoder.predict(X_train)
print "INFO: ...done."
print "INFO: decoding encoded input..."
y_decoded = decoder.predict(y_encoded)
print "INFO: ..done"


X_train = scaler.inverse_transform(X_train)
X_decoded = scaler.inverse_transform(y_decoded)

encoder_filename = "lorentz/model_encoder.%s.h5" % (level)
decoder_filename = "lorentz/model_decoder.%s.h5" % (level)

encoder.save(encoder_filename)
decoder.save(decoder_filename)

print "INFO: encoder saved to file:", encoder_filename
print "INFO: decoder saved to file:", decoder_filename

for i in range(20):
    # (pT, eta, phi, E, M)

    x1 = TLorentzVector()
    x2 = TLorentzVector()
    xx = TLorentzVector()

    x1.SetPtEtaPhiM(X_train[i][0],
                    X_train[i][1],
                    0.,
                    X_train[i][2])

    x2.SetPtEtaPhiM(X_train[i][3],
                    X_train[i][4],
                    X_train[i][5],
                    X_train[i][6])

    xx.SetPtEtaPhiM(X_train[i][7],
                    X_train[i][8],
                    X_train[i][9],
                    X_train[i][10])
    xx.dEta = X_train[i][11]
    xx.dPhi = X_train[i][12]
    xx.dR = X_train[i][13]

    y1 = TLorentzVector()
    y2 = TLorentzVector()
    yy = TLorentzVector()

    y1.SetPtEtaPhiM(X_decoded[i][0],
                    X_decoded[i][1],
                    0.,
                    X_decoded[i][2])

    y2.SetPtEtaPhiM(X_decoded[i][3],
                    X_decoded[i][4],
                    X_decoded[i][5],
                    X_decoded[i][6])

    yy.SetPtEtaPhiM(X_decoded[i][7],
                    X_decoded[i][8],
                    X_decoded[i][9],
                    X_decoded[i][10])
    yy.dEta = X_decoded[i][11]
    yy.dPhi = X_decoded[i][12]
    yy.dR = X_decoded[i][13]

    print "original: (%.1f, %.2f, %.2f, %.1f) :: (%.1f, %.2f, %.2f, %.1f) :: (%.1f, %.2f, %.2f, %.1f) :: (%.2f, %.2f, %.2f)" % (
        x1.Pt(), x1.Eta(), x1.Phi(), x1.M(),
        x2.Pt(), x2.Eta(), x2.Phi(), x2.M(),
        xx.Pt(), xx.Eta(), xx.Phi(), xx.M(),
        xx.dEta, xx.dPhi, xx.dR)
    print "decoded: (%.1f, %.2f, %.2f, %.1f) :: (%.1f, %.2f, %.2f, %.1f) :: (%.1f, %.2f, %.2f, %.1f) :: (%.2f, %.2f, %.2f)" % (
        y1.Pt(), y1.Eta(), y1.Phi(), y1.M(),
        y2.Pt(), y2.Eta(), y2.Phi(), y2.M(),
        yy.Pt(), yy.Eta(), yy.Phi(), yy.M(),
        yy.dEta, yy.dPhi, yy.dR)
    print "-"*20
