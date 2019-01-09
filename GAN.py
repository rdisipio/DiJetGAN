#!/usr/bin/env python

import os, sys
import csv
import argparse
import random

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
np.set_printoptions( precision=2, suppress=True, linewidth=200 )

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from keras.utils import to_categorical
#from keras.utils import multi_gpu_model

from keras.optimizers import *
from keras import regularizers
from keras.callbacks import *

import pandas as pd

############

known_classifiers = [ "rnn:GAN", "rnn:highlevel", "rnn:PxPyPzMBwNtrk" ]

parser = argparse.ArgumentParser(description='ttbar diffxs sqrt(s) = 13 TeV classifier training')
parser.add_argument( '-i', '--training_filename', default="" )
parser.add_argument( '-c', '--classifier',        default=known_classifiers[0] )
parser.add_argument( '-p', '--preselection',      default="incl" )
parser.add_argument( '-s', '--systematic',        default="nominal" ) 
parser.add_argument( '-d', '--dsid',              default="361024" )
parser.add_argument( '-e', '--epochs',            default=1000 )
args         = parser.parse_args()

classifier        = args.classifier
training_filename = args.training_filename
preselection      = args.preselection
systematic        = args.systematic
dsid              = args.dsid
n_epochs          = int(args.epochs)

classifier_arch, classifier_feat = classifier.split(':')

if training_filename == "":
#   training_filename = "csv/training.%s.%s.%s.%s.csv" % ( classifier_arch, classifier_feat, preselection, systematic )
   training_filename = "csv/mc16a.%s.%s.%s.%s.%s.csv" % ( dsid, classifier_arch, classifier_feat, preselection, systematic )
   print "INFO: training file:", training_filename
else:
   systematic = training_filename.split("/")[-1].split('.')[-2]

print "INFO: training systematic: %s" % systematic

scaler = MinMaxScaler( (-1,1) )

from features import *

#features = [
#   "ljet1_px", "ljet1_py", "ljet1_pz", "ljet1_M", #"ljet1_tau32",
#   "ljet2_px", "ljet2_py", "ljet2_pz", "ljet2_M", #"ljet2_tau32",
#]

#features = [
#   "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
#   "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
#   "jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M"
#   "jj_pt",    "jj_eta",    "jj_phi",    "jj_M",    "jj_dPhi", "jj_dR",
#]

features = [
   "ljet1_pt", "ljet1_eta", "ljet1_pz", "ljet1_E", "ljet1_M",
   "ljet2_pt", "ljet2_eta", "ljet2_pz", "ljet2_E", "ljet2_M",
   "jj_pt",    "jj_eta",    "jj_pz",    "jj_E",    "jj_M",
   "jj_dPhi",  "jj_dEta",   "jj_dR",
   ]

n_features = len(features)
print "INFO: input features:"
print features
print "INFO: total number of input features:     ", n_features

# read in input file
data = pd.read_csv( training_filename, delimiter=',', names=header )
print "INFO: dataset loaded into memory"
print "INFO: header:"
print header

#print data.isnull().values.any()
print "INFO: checking if input data has NaNs"
nan_rows = data[data.isnull().T.any().T]
print nan_rows
data.dropna(inplace=True)

print "INFO: number of good events:", len(data)

X_train = data[features].values

print "INFO: X_train before standardization:"
print X_train

X_train = scaler.fit_transform( X_train )
print "INFO: X_train after standardization:"
print X_train

n_events = len( X_train )
print "INFO: number of training events:", n_events

print "INFO: X_train shape:", X_train.shape
print X_train

# Use MC event weights for training?
event_weights = None
#event_weights = data["weight"].values
print "INFO: event weights:"
print event_weights

#~~~~~~~~~~~~~~~~~~~~~~

from models import *

def make_generator():
   #return make_generator_mlp_PtEtaPhiM( GAN_noise_size )
   #return make_generator_mlp_PxPyPzE( GAN_noise_size )
   return make_generator_mlp( GAN_noise_size, n_features )
   #return make_generator_rnn( GAN_noise_size, n_features )
   #return make_generator_cnn( GAN_noise_size, n_features )

def make_discriminator():
   #return make_discriminator_mlp( n_features )
   #return make_discriminator_rnn( n_features )
   return make_discriminator_cnn( n_features )

#~~~~~~~~~~~~~~~~~~~~~~

GAN_noise_size = 128 # number of random numbers (input noise)

d_optimizer   = Adam(0.001, 0.5) #(0.0001, 0.5)
g_optimizer   = Adam(0.001, 0.5) #(0.0001, 0.5)
#d_optimizer  = Adam(0.0001)
#g_optimizer  = Adam(0.0001)

discriminator = make_discriminator()
discriminator.name = "Discriminator"
discriminator.compile( loss='binary_crossentropy',
                       optimizer=d_optimizer,
                       metrics=['accuracy'] )
discriminator.summary()

generator = make_generator()
generator.name = "Generator"
generator.compile( loss='mean_squared_error',
                   optimizer=g_optimizer )
generator.summary()

# For the combined model we will only train the generator
discriminator.trainable = False

GAN_input  = Input( shape=(GAN_noise_size,) )
GAN_hidden = generator(GAN_input)
GAN_output = discriminator(GAN_hidden)
GAN = Model( GAN_input, GAN_output )
GAN.name = "GAN"
GAN.compile( loss='binary_crossentropy',
             optimizer=g_optimizer )
GAN.summary()

# Training:
# 1) pick up ntrain events from real dataset
# 2) generate ntrain fake events

# Pre-train discriminator
ntrain = 10000
train_idx = random.sample( range(0,X_train.shape[0]), ntrain)
X_train_real = X_train[train_idx,:]

X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_noise_size])
X_train_fake = generator.predict(X_noise)

# create GAN training dataset
X = np.concatenate( (X_train_real, X_train_fake) )
n = X_train_real.shape[0]
y = np.zeros([2*n])
y[:n] = 1
y[n:] = 0

print "INFO: pre-training discriminator network"
discriminator.trainable = True
discriminator.fit(X,y, epochs=1, batch_size=128)
y_hat = discriminator.predict(X)

# set up loss storage vector
history = { "d_loss":[], "d_loss_r":[], "d_loss_f":[], "g_loss":[], "d_acc":[], "g_acc":[] }

#######################

def train_loop(nb_epoch=1000, BATCH_SIZE=32):

   plt_frq = max( 1, int(nb_epoch)/20 )

   y_real = np.ones(  (BATCH_SIZE,1) )
   y_fake = np.zeros( (BATCH_SIZE,1) )

   for epoch in range(nb_epoch):

        # select some real events
        train_idx = np.random.randint( 0, X_train.shape[0], size=BATCH_SIZE )
        X_train_real = X_train[train_idx,:]

        # generate fake events
        X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_noise_size])
        X_train_fake = generator.predict(X_noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real, d_acc_real = discriminator.train_on_batch( X_train_real, y_real )
        d_loss_fake, d_acc_fake = discriminator.train_on_batch( X_train_fake, y_fake )
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc  = 0.5 * np.add(d_acc_real, d_acc_fake)

        history["d_loss"].append(d_loss)
        history["d_loss_r"].append(d_loss_real)
        history["d_loss_f"].append(d_loss_fake)
        history["d_acc"].append(d_acc)

        # Train the generator
        # create new (statistically independent) random noise sample
        #X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_noise_size])
        X_train_fake = generator.predict(X_noise)

        # we want discriminator to mistake images as real
        g_loss = GAN.train_on_batch( X_noise, y_real )
        history["g_loss"].append(g_loss)

        if epoch % plt_frq == 0:
           print "Epoch: %5i :: d_loss = %.3f ( real = %.3f, fake = %.3f ) :: g_loss = %.3f" % ( epoch, d_loss, d_loss_real, d_loss_fake, g_loss )

   return history
 
#######################

print "INFO: Train for %i epochs" % ( n_epochs )
train_loop( nb_epoch=n_epochs, BATCH_SIZE=128 )

# save model to file
model_filename = "GAN/generator.%s.%s.%s.%s.%s.h5" % (dsid,classifier_arch, classifier_feat, preselection, systematic)
generator.save( model_filename )
print "INFO: generator model saved to file", model_filename

scaler_filename = "GAN/scaler.%s.%s.%s.%s.%s.pkl" % (dsid,classifier_arch, classifier_feat, preselection, systematic)
with open( scaler_filename, "wb" ) as file_scaler:
   pickle.dump( scaler, file_scaler )
print "INFO: scaler saved to file", scaler_filename

from ROOT import *
gROOT.SetBatch(1)

training_root = TFile.Open( "GAN/training_history.%s.%s.%s.%s.%s.root" % (dsid, classifier_arch, classifier_feat, preselection, systematic), "RECREATE" )
print "INFO: saving training history..."

#h_d_loss = TH1F( "d_loss", ";Epoch;Discriminator Loss",     n_epochs, 0.5, n_epochs+0.5 )
#h_d_acc  = TH1F( "d_acc",  ";Epoch;Discriminator Accuracy", n_epochs, 0.5, n_epochs+0.5 )
#h_g_loss = TH1F( "g_loss", ";Epoch;Generator loss",         n_epochs, 0.5, n_epochs+0.5 )
#h_g_acc  = TH1F( "g_acc",  ";Epoch;Generator Accuracy",     n_epochs, 0.5, n_epochs+0.5 )

h_d_loss   = TGraphErrors()
h_d_loss_r = TGraphErrors()
h_d_loss_f = TGraphErrors()
h_d_acc    = TGraphErrors()
h_g_loss   = TGraphErrors()
h_g_acc    = TGraphErrors()

h_d_loss.SetLineColor(kRed)
h_g_loss.SetLineColor(kBlue)

n_epochs = len(history['d_loss'])
for i in range( n_epochs ):
      d_loss = history['d_loss'][i]
      d_loss_r = history['d_loss_r'][i]
      d_loss_f = history['d_loss_f'][i]
      d_acc  = history['d_acc'][i]
      g_loss = history['g_loss'][i]
      #h_d_loss.SetBinContent( i+1, d_loss )
      #h_d_acc.SetBinContent( i+1,  d_acc )
      #h_g_loss.SetBinContent( i+1, g_loss )
      h_d_loss.SetPoint( i, i, d_loss )
      h_d_loss_r.SetPoint( i, i, d_loss_r )
      h_d_loss_f.SetPoint( i, i, d_loss_f )
      h_d_acc.SetPoint( i, i, d_acc )
      h_g_loss.SetPoint( i, i, g_loss )
h_d_loss.Write( "d_loss" )
h_d_loss_r.Write( "d_loss_r" )
h_d_loss_f.Write( "d_loss_f" )
h_d_acc.Write( "d_acc" )
h_g_loss.Write( "g_loss")

training_root.Write()
training_root.Close()
print "INFO: training history saved to file:", training_root.GetName()
