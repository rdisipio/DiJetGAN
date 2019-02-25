#!/usr/bin/env python

import os
import sys
import csv
import argparse
import random
import math

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=300)

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# from keras.utils import to_categorical
# from keras.utils import multi_gpu_model

from keras.optimizers import *
from keras import regularizers
from keras.callbacks import *
from keras.utils import plot_model

import pandas as pd
import helper_functions as hf
# from fourmomentum_scaler import *

from ROOT import *
gROOT.SetBatch(1)

############

parser = argparse.ArgumentParser(
    description='ttbar diffxs sqrt(s) = 13 TeV classifier training')
parser.add_argument('-i', '--training_filename', default="")
parser.add_argument('-l', '--level',             default="reco")
parser.add_argument('-p', '--preselection',      default="pt250")
parser.add_argument('-s', '--systematic',        default="nominal")
parser.add_argument('-d', '--dsid',              default="mg5_dijet_ht500")
parser.add_argument('-e', '--epochs',            default=1000)
args = parser.parse_args()

training_filename = args.training_filename
level = args.level
preselection = args.preselection
systematic = args.systematic
dsid = args.dsid
n_epochs = int(args.epochs)

if training_filename == "":
    #   training_filename = "csv/training.%s.%s.%s.%s.csv" % ( classifier_arch, classifier_feat, preselection, systematic )
    training_filename = "csv/%s.%s.%s.%s.csv" % (
        dsid, level, preselection, systematic)
    print "INFO: training file:", training_filename
else:
    systematic = training_filename.split("/")[-1].split('.')[-2]

print "INFO: training level: %s" % level
print "INFO: training systematic: %s" % systematic

from features import *

if level == "ptcl":
    # features = [
    #    "ljet1_pt", "ljet1_eta", "ljet1_M",
    #    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M",
    #]
    features = [
        "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
        "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
        "jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M",
        "jj_dPhi",  "jj_dEta",  "jj_dR",
    ]
else:
    features = [
        "ljet1_pt", "ljet1_eta", "ljet1_M",
        "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M",
        "mu",
    ]


n_features = len(features)
print "INFO: input features:"
print features
print "INFO: total number of input features:     ", n_features

# read in input file
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

print "INFO: X_train before standardization:"
print X_train

# load scaler
scaler_filename = "GAN_%s/scaler.%s.pkl" % (level, level)
print "INFO: loading scaler from", scaler_filename
with open(scaler_filename, "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

# Use MC event weights for training?
# print X_train[:,10]
# j_pt_max = 1000.
# event_weights = np.array( [ x/j_pt_max if not x == 0. else 1./j_pt_max for x in X_train[:,10] ] )
event_weights = None
# event_weights = data["weight"].values
print "INFO: event weights:"
print event_weights

X_train = scaler.transform(X_train)
print "INFO: X_train after standardization:"
print X_train

n_events = len(X_train)
print "INFO: number of training events:", n_events

#~~~~~~~~~~~~~~~~~~~~~~

from models import *

GAN_noise_size = 128  # number of random numbers (input noise)
G_output_size = 8
# n_features = 18 (?)

encoder = make_encoder(n_features, G_output_size)
encoder.name = "Encoder"
decoder = make_decoder(G_output_size, n_features)
decoder.name = "Decoder"


def make_generator():
    # return make_generator_mlp_LorentzVector( GAN_noise_size )
    # return make_generator_mlp(GAN_noise_size, G_output_size )
    # return make_generator_rnn( GAN_noise_size, G_output_size )
    return make_generator_cnn(GAN_noise_size, n_features)


def make_discriminator():
    # return make_discriminator_mlp( G_output_size )
    # return make_discriminator_rnn( G_output_size )
    return make_discriminator_cnn(G_output_size)

#~~~~~~~~~~~~~~~~~~~~~~


# d_optimizer = RMSprop(lr=1e-4, rho=0.9)  # clipvalue=0.01)
# g_optimizer = RMSprop(lr=1e-4, rho=0.9)  # , clipvalue=0.01)

# d_optimizer = Adamax()
# g_optimizer = Adadelta()

d_optimizer = Adam(1e-5, beta_1=0.5, beta_2=0.9)
g_optimizer = Adam(1e-5, beta_1=0.5, beta_2=0.9)

#d_optimizer = Adam(1e-5, beta_1=0.8, beta_2=0.999 )
#g_optimizer = Adam(1e-5, beta_1=0.8, beta_2=0.999 )

# d_optimizer = Adam(0.0001)  # , clipnorm=1.0)
# g_optimizer = Adam(0.0001)  # , clipnorm=1.0)

# d_optimizer = Adam(0.0001)  # , 0.5)
# g_optimizer = Adam(0.0001) #, 0.5)

# d_optimizer = Adam(1e-4)
# g_optimizer = Adam(1e-4)

# d_optimizer = SGD(0.0001, 0.9, nesterov=True)
# g_optimizer = SGD(0.0001, 0.9, nesterov=True)

# d_optimizer = SGD(0.01, 0.9)
# g_optimizer = SGD(0.01, 0.9)


# d_optimizer = Adam(0.01)
# g_optimizer = Adam(0.01)

# the best so far, and by far!
#d_optimizer = SGD(0.01)
#g_optimizer = SGD(0.01)

# d_optimizer = Adam(0.001, 0.9)
# g_optimizer = Adam(0.001, 0.9)

# d_optimizer = RMSprop(lr=0.01)
# g_optimizer = SGD(0.01)

###########
# Generator
###########
generator = make_generator()
generator.name = "Generator"
generator.compile(
    loss='mean_squared_error',
    optimizer=g_optimizer)
print "INFO: Generator:"
generator.summary()

###############
# Discriminator
###############

#discriminator = make_discriminator()
#discriminator.name = "Discriminator"
# discriminator.compile(
#    loss='binary_crossentropy',
#    optimizer=d_optimizer,
#    metrics=['accuracy'])
#print "INFO: Discriminator:"
# discriminator.summary()

# For the combined model we will only train the generator
#discriminator.trainable = False
#GAN_input = Input(shape=(GAN_noise_size,))
#GAN_latent = generator(GAN_input)
#GAN_output = discriminator(GAN_latent)
#GAN = Model(GAN_input, GAN_output)
#GAN.name = "GAN"
# GAN.compile(
#    loss='binary_crossentropy',
#    optimizer=g_optimizer)
#print "INFO: GAN:"
# GAN.summary()

discriminator = make_discriminator()
discriminator.name = "Discriminator"

# autoencoder
ae_in = Input((n_features,))  # (18)
ae_hid = encoder(ae_in)        # (8)
D_out = discriminator(ae_hid)  # (1)
ae_out = decoder(ae_hid)       # (18)
AE = Model(ae_in, [ae_out, D_out])  # (18), (18,1)
AE.name = "Autoencoder"
AE.compile(
    loss=['mse', 'binary_crossentropy'],
    optimizer=d_optimizer,
    metrics=['accuracy']
)
print "INFO: Autoencoder:"
AE.summary()

discriminator.trainable = False
GAN_input = Input(shape=(GAN_noise_size,))  # (128)
GAN_latent = encoder(generator(GAN_input))  # (18)->(8)
GAN_output = discriminator(GAN_latent)  # (1)
GAN = Model(GAN_input, GAN_output)  # (18), (1)
GAN.name = "GAN"
GAN.compile(
    loss='binary_crossentropy',
    optimizer=g_optimizer)
print "INFO: GAN:"
GAN.summary()


print "INFO: saving models to png files"
if not os.path.exists("img/"):
    os.makedirs("img/")

plot_model(generator,      show_shapes=True,
           to_file="img/DCGAN_model_%s_generator.png" % (dsid))
plot_model(discriminator,  show_shapes=True,
           to_file="img/DCGAN_model_%s_discriminator.png" % (dsid))
plot_model(GAN,            show_shapes=True,
           to_file="img/DCGAN_model_%s_GAN.png" % (dsid))
plot_model(AE,            show_shapes=True,
           to_file="img/DCGAN_model_%s_AE.png" % (dsid))

# Training:
# 1) pick up ntrain events from real dataset
# 2) generate ntrain fake events

# Pre-train discriminator
ntrain = 20000
train_idx = random.sample(range(0, X_train.shape[0]), ntrain)
X_train_real = X_train[train_idx, :]

X_noise = np.random.uniform(0, 1, size=[X_train_real.shape[0], GAN_noise_size])
# X_noise = np.random.uniform(-1,1,size=[X_train_real.shape[0], GAN_noise_size])
# X_noise = np.random.normal(0., 1., (X_train_real.shape[0], GAN_noise_size))
X_train_fake = generator.predict(X_noise)

# create GAN training dataset
X = np.concatenate((X_train_real, X_train_fake))
n = X_train_real.shape[0]
y = np.zeros([2*n])
y[:n] = 1
y[n:] = 0

# event weights
# weights_fake = np.array( [ x if not x == 0. else 1./j_pt_max for x in X_train_fake[:,10] ] )
# weights_real = event_weights[train_idx]
# weights = np.concatenate( (weights_real, weights_fake) )

# print "INFO: pre-training discriminator network"
discriminator.trainable = True
#discriminator.fit(X, y, epochs=1, batch_size=128)
AE.fit(X, [X, y], epochs=1, batch_size=128)

history = {
    "d_lr": [], "g_lr": [],
    "d_loss": [], "d_loss_r": [], "d_loss_f": [],
    "g_loss": [],
    "d_acc": [], "d_acc_r": [], "d_acc_f": [],
}

#######################

# learning rate schedule


def step_decay(epoch, initial_lrate=0.01, drop=0.5, epochs_drop=10.0):
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def train_loop(nb_epoch=1000, BATCH_SIZE=32, TRAINING_RATIO=1):
    global epoch_overall

    print "INFO: Train for %i epochs with BATCH_SIZE=%i and TRAINING_RATIO=%i" % (
        n_epochs, BATCH_SIZE, TRAINING_RATIO)

    plt_frq = max(1, int(nb_epoch)/50)

    y_real = np.ones((BATCH_SIZE, 1))
    y_fake = np.zeros((BATCH_SIZE, 1))

    # d_lr_0 = float( K.get_value( discriminator.optimizer.lr ) )
    # g_lr_0 = float( K.get_value( generator.optimizer.lr ) )

    for epoch in range(nb_epoch):

        d_lr = float(K.get_value(AE.optimizer.lr))
        history['d_lr'].append(d_lr)

        g_lr = float(K.get_value(AE.optimizer.lr))
        history['g_lr'].append(g_lr)

#        d_lr = step_decay( epoch, initial_lrate=d_lr_0, drop=0.5, epochs_drop=nb_epoch/10.)
#        d_decay = d_lr_0 / float(nb_epoch)
#        d_lr = d_lr / (1. + d_decay * epoch)
#        K.set_value(discriminator.optimizer.lr, d_lr)

#        g_lr = step_decay( epoch, initial_lrate=g_lr_0, drop=0.5, epochs_drop=nb_epoch/10.)
#        g_decay = g_lr_0 / float(nb_epoch)
#        g_lr = g_lr / (1. + g_decay * epoch)
#        K.set_value(generator.optimizer.lr, g_lr)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        for _ in range(TRAINING_RATIO):
            # select some real events
            train_idx = np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)
            X_train_real = X_train[train_idx, :]

            # generate fake events
            X_noise = np.random.uniform(
                0, 1, size=[BATCH_SIZE, GAN_noise_size])
            # X_noise = np.random.normal(0., 1, (BATCH_SIZE, GAN_noise_size))
            X_train_fake = generator.predict(X_noise)

            discriminator.trainable = True

            loss_r, dec_loss_r, d_loss_r, dec_acc_r, d_acc_r = AE.train_on_batch(
                X_train_real, [X_train_real, y_real])
            loss_f, dec_loss_f, d_loss_f, dec_acc_f, d_acc_f = AE.train_on_batch(
                X_train_fake, [X_train_fake, y_fake])

            #clip_weights(discriminator, 0.01)

        d_loss = 0.5 * np.add(d_loss_r, d_loss_f)

        d_acc = 0.5 * np.add(d_acc_r, d_acc_f)

        history["d_loss"].append(d_loss)
        history["d_loss_r"].append(d_loss_r)
        history["d_loss_f"].append(d_loss_f)
        history["d_acc"].append(d_acc)
        history["d_acc_f"].append(d_acc_f)
        history["d_acc_r"].append(d_acc_r)

        # ---------------------
        #  Train Generator
        # ---------------------

        # we want discriminator to mistake images as real
        discriminator.trainable = False
        encoder.trainable = False

        g_loss = GAN.train_on_batch(X_noise, y_real)
        history["g_loss"].append(g_loss)

        if epoch % plt_frq == 0:
            print "Epoch: %5i/%5i :: BS = %i, d_lr = %.5f, g_lr = %.5f :: d_loss = %.2f ( real = %.2f, fake = %.2f ), d_acc = %.2f ( real = %.2f, fake = %.2f ), g_loss = %.2f" % (
                epoch, nb_epoch, BATCH_SIZE, d_lr, g_lr, d_loss, d_loss_r, d_loss_f, d_acc, d_acc_r, d_acc_f, g_loss)

            model_filename = "GAN_%s/DCGAN.generator.%s.%s.%s.%s.epoch_%05i.h5" % (
                level, dsid, level, preselection, systematic, epoch_overall)
            generator.save(model_filename)

            training_filename = "GAN_%s/DCGAN.training_history.%s.%s.%s.%s.root" % (
                level, dsid, level, preselection, systematic)
            hf.save_training_history(history, training_filename, verbose=False)

        epoch_overall += 1

    return history

#######################


if not os.path.exists("GAN_%s/" % level):
    os.makedirs("GAN_%s/" % level)

epoch_overall = 0

train_loop(nb_epoch=n_epochs, BATCH_SIZE=32,  TRAINING_RATIO=1)

# train_loop(nb_epoch=n_epochs, BATCH_SIZE=32,  TRAINING_RATIO=5)
# train_loop(nb_epoch=n_epochs, BATCH_SIZE=64,  TRAINING_RATIO=1)
#train_loop(nb_epoch=n_epochs, BATCH_SIZE=128, TRAINING_RATIO=1)
# train_loop(nb_epoch=n_epochs/10, BATCH_SIZE=512, TRAINING_RATIO=1)
# train_loop( nb_epoch=n_epochs, BATCH_SIZE=512 )
# train_loop(nb_epoch=int(n_epochs/2), BATCH_SIZE=1024, TRAINING_RATIO=1)

# lr = float( K.get_value( discriminator.optimizer.lr ) )
# K.set_value(discriminator.optimizer.lr, 0.001)
# K.set_value(generator.optimizer.lr, 0.001)
# train_loop(nb_epoch=n_epochs/2, BATCH_SIZE=32,  TRAINING_RATIO=1)

# K.set_value(discriminator.optimizer.lr, 0.0001)
# K.set_value(generator.optimizer.lr, 0.0001)
# train_loop(nb_epoch=n_epochs/2, BATCH_SIZE=32,  TRAINING_RATIO=1)

# save model to file
model_filename = "GAN/%s/DCGAN.generator.%s.%s.%s.%s.h5" % (
    level, dsid, level, preselection, systematic)
generator.save(model_filename)
print "INFO: generator model saved to file", model_filename

training_filename = "GAN/DCGAN.training_history.%s.%s.%s.%s.root" % (
    dsid, level, preselection, systematic)
hf.save_training_history(history, training_filename)