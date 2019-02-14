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

from keras.optimizers import *
from keras import regularizers
from keras.callbacks import *
from keras.utils import plot_model

from models import *

import pandas as pd
import helper_functions as hf

from ROOT import *
gROOT.SetBatch(1)

parser = argparse.ArgumentParser(
    description='ttbar diffxs sqrt(s) = 13 TeV classifier training')
parser.add_argument('-i', '--training_filename', default="")
parser.add_argument('-l', '--level',             default="ptcl")
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

features = [
    "ljet1_pt", "ljet1_eta", "ljet1_M",
    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M"
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
scaler_filename = "GAN/scaler.%s.pkl" % level
print "INFO: loading scaler from", scaler_filename
with open(scaler_filename, "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

X_train = scaler.transform(X_train)
print "INFO: X_train after standardization:"
print X_train

n_events = len(X_train)
print "INFO: number of training events:", n_events

from models import make_generator_cnn, make_discriminator_cnn

GAN_noise_size = 128  # number of random numbers (input noise)

d_optimizer = Adam(1e-4, beta_1=0.5, beta_2=0.9)
g_optimizer = Adam(1e-4, beta_1=0.5, beta_2=0.9)

generator = make_generator_cnn(GAN_noise_size, n_features)
generator.name = "Generator"
generator.compile(
    loss='mse',
    optimizer=g_optimizer)
print "INFO: generator:"
generator.summary()

discriminator = make_discriminator_cnn(n_features)
discriminator.name = "Discriminator"
print "INFO: discriminator:"
discriminator.summary()

# Build GAN
discriminator.trainable = False
GAN_input = Input(shape=(GAN_noise_size,))
GAN_latent = generator(GAN_input)
GAN_output = discriminator(GAN_latent)
GAN = Model(GAN_input, GAN_output)
GAN.name = "GAN"
GAN.compile(
    loss=wasserstein_loss,
    optimizer=g_optimizer)
print "INFO: GAN:"
GAN.summary()

# Build Discriminator w/ gradient

shape = discriminator.get_input_shape_at(0)[1:]
D_input_fake = Input(shape=shape, name="in_fake")
D_input_real = Input(shape=shape, name="in_real")
D_input_grad = Input(shape=shape, name="in_grad")
diff = subtract([discriminator(D_input_fake),
                 discriminator(D_input_real)], name="diff")
norm = GradNorm(name="gradient")([discriminator(D_input_grad), D_input_grad])
discriminator_grad = Model(
    inputs=[D_input_fake, D_input_real, D_input_grad], outputs=[diff, norm])
discriminator.trainable = True
lmbd = 10.
discriminator_grad.compile(
    loss=[mean_loss, 'mse'],
    loss_weights=[1.0, lmbd],
    optimizer=d_optimizer,
    metrics=['accuracy']
)
print "INFO: discriminator w/ gradient:"
discriminator_grad.summary()

print "INFO: saving models to png files"
if not os.path.exists("img/"):
    os.makedirs("img/")
plot_model(generator,      show_shapes=True,
           to_file="img/model_WGANGP_%s_generator.png" % (dsid))
plot_model(discriminator,  show_shapes=True,
           to_file="img/model_WGANGP_%s_discriminator.png" % (dsid))
plot_model(GAN,            show_shapes=True,
           to_file="img/model_WGANGP_%s_GAN.png" % (dsid))

history = {
    "d_lr": [], "g_lr": [],
    "d_loss": [],
    "g_loss": [],
    "d_acc": [],
}

if not os.path.exists("GAN/"):
    os.makedirs("GAN/")


def train_loop(nb_epoch=1000, BATCH_SIZE=32, TRAINING_RATIO=1):
    global epoch_overall

    print "INFO: Train for %i epochs with BATCH_SIZE=%i and TRAINING_RATIO=%i" % (
        n_epochs, BATCH_SIZE, TRAINING_RATIO)

    plt_frq = max(1, int(nb_epoch)/20)

    y_real = -1 * np.ones((BATCH_SIZE, 1))
    y_fake = np.ones((BATCH_SIZE, 1))

    for epoch in range(nb_epoch):

        d_lr = float(K.get_value(discriminator_grad.optimizer.lr))
        history['d_lr'].append(d_lr)

        g_lr = float(K.get_value(generator.optimizer.lr))
        history['g_lr'].append(g_lr)

        # ---------------------
        #  Train Generator
        # ---------------------

        # generate fake events
        X_noise = np.random.uniform(
            0, 1, size=[BATCH_SIZE, GAN_noise_size]).astype('float32')

        g_loss = GAN.train_on_batch(X_noise, y_real)
        g_loss = -1*g_loss
        history["g_loss"].append(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # select some real events
        train_idx = np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)
        X_train_real = X_train[train_idx, :]

        X_train_fake = generator.predict(X_noise)

        epsilon = np.random.uniform(0, 1, size=(BATCH_SIZE, 1))
        X_train_grad = epsilon*X_train_real + (1-epsilon)*X_train_fake

        #print X_train_fake.shape
        #print X_train_real.shape
        #print X_train_grad.shape

        y_identity = np.ones((BATCH_SIZE, 1))

        d_loss, d_loss_diff, d_loss_norm, d_acc_diff, d_acc_norm = discriminator_grad.train_on_batch(
            [X_train_fake, X_train_real, X_train_grad],
            [y_identity, y_identity])

        d_acc = 0.5*(d_acc_diff + d_acc_norm)

        history['d_loss'].append(d_loss)
        history['d_acc'].append(d_acc)

        if epoch % plt_frq == 0:
            print "Epoch: %5i/%5i :: BS = %i, d_lr = %.5f, g_lr = %.5f :: d_loss = %.2f, d_acc = %.2f, g_loss = %.2f" % (
                epoch, nb_epoch, BATCH_SIZE, d_lr, g_lr,
                d_loss, d_acc, g_loss)

            model_filename = "GAN/WGANGP.generator.%s.%s.%s.%s.epoch_%05i.h5" % (
                dsid, level, preselection, systematic, epoch_overall)
            generator.save(model_filename)

            # print "H(d) = %f : H(g) = %f" % (
            #    entropy(discriminator), entropy(generator))
        epoch_overall += 1

    return history

#######################


epoch_overall = 0

train_loop(nb_epoch=n_epochs, BATCH_SIZE=32,  TRAINING_RATIO=1)

# save model to file
model_filename = "GAN/WGANGP.generator.%s.%s.%s.%s.h5" % (
    dsid, level, preselection, systematic)
generator.save(model_filename)
print "INFO: generator model saved to file", model_filename

training_root = TFile.Open("GAN/WGANGP.training_history.%s.%s.%s.%s.root" % (
    dsid, level, preselection, systematic), "RECREATE")
print "INFO: saving training history..."

h_d_lr = TGraphErrors()
h_g_lr = TGraphErrors()
h_d_loss = TGraphErrors()
h_d_acc = TGraphErrors()
h_g_loss = TGraphErrors()

n_epochs = len(history['d_loss'])
for i in range(n_epochs):
    d_lr = history['d_lr'][i]
    g_lr = history['g_lr'][i]
    d_loss = history['d_loss'][i]
    d_acc = history['d_acc'][i]
    g_loss = history['g_loss'][i]

    h_d_lr.SetPoint(i, i, d_lr)
    h_g_lr.SetPoint(i, i, g_lr)
    h_d_loss.SetPoint(i, i, d_loss)
    h_d_acc.SetPoint(i, i, d_acc)
    h_g_loss.SetPoint(i, i, g_loss)

h_d_lr.Write("d_lr")
h_g_lr.Write("g_lr")
h_d_loss.Write("d_loss")
h_g_loss.Write("g_loss")
h_d_acc.Write("d_acc")

training_root.Write()
training_root.Close()
print "INFO: training history saved to file:", training_root.GetName()
