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
from keras.utils import plot_model

import pandas as pd

############
        
known_classifiers = [ "rnn:GAN", "rnn:highlevel", "rnn:PxPyPzMBwNtrk" ]

parser = argparse.ArgumentParser(description='ttbar diffxs sqrt(s) = 13 TeV classifier training')
parser.add_argument( '-i', '--training_filename', default="" )
parser.add_argument( '-c', '--classifier',        default=known_classifiers[0] )
parser.add_argument( '-p', '--preselection',      default="incl" )
parser.add_argument( '-s', '--systematic',        default="nominal" ) 
parser.add_argument( '-d', '--dsid',              default="mg5_dijet_ht500" )
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

features = [
   "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
   "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
   "jj_pt",    "jj_eta",    "jj_phi", "jj_E", "jj_M",
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
   #return make_generator_mlp_LorentzVector( GAN_noise_size )
   return make_generator_mlp( GAN_noise_size, n_features )
   #return make_generator_rnn( GAN_noise_size, n_features )
   #return make_generator_cnn( GAN_noise_size, n_features )

def make_discriminator():
   #return make_discriminator_mlp( n_features )
   #return make_discriminator_rnn( n_features )
   return make_discriminator_cnn( n_features )

#~~~~~~~~~~~~~~~~~~~~~~

GAN_noise_size = 128 # number of random numbers (input noise)

d_optimizer   = Adam(0.0001, beta_1=0.5, beta_2=0.9)
g_optimizer   = Adam(0.0001, beta_1=0.5, beta_2=0.9)

#d_optimizer  = Adam(0.0001,  )
#g_optimizer  = Adam(0.0001,  )

#d_optimizer  = Adam(0.001,0.7)
#g_optimizer  = Adam(0.001,0.7)

#d_optimizer = SGD(0.001, 0.9 ) #, nesterov=True)
#g_optimizer = SGD(0.001, 0.9 ) #, nesterov=True)

#d_optimizer = SGD()
#g_optimizer = SGD()

#d_optimizer = Adam()
#g_optimizer = Adam()

###########
# Generator
###########
generator = make_generator()
generator.name = "Generator"
generator.compile( loss='mean_squared_error',
                   optimizer=g_optimizer )
generator.summary()

###############
# Discriminator
###############

#D_orig = make_discriminator()
#D_orig.name = "Discr_orig"
#D_flip = make_discriminator()
#D_flip.name = "Discr_flip"
#D_input_orig  = Input( shape=(n_features,), name='D_input' )
#D_input_flip  = Lambda( flip_eta, name="Eta_flip" )(D_input_orig)
#D_output_orig = D_orig(D_input_orig)
#D_output_flip = D_flip(D_input_flip)

#D = make_discriminator()
#D.name = "Discr"
#D_input_orig  = Input( shape=(n_features,), name='D_input_orig' )
#D_input_flip  = Input( shape=(n_features,), name='D_input_flip' )
#D_output_flip  = Lambda( flip_eta, name="Eta_flip" )(D_input_flip)
#D_output_orig = D(D_input_orig)
#D_output_flip = D(D_output_flip)

#D_output_orig = Dense( 1, activation="sigmoid", name="output_orig")(D_output_orig)
#D_output_flip = Dense( 1, activation="sigmoid", name="output_flip")(D_output_flip)
#discriminator = Model( D_input_orig, [D_output_orig, D_output_flip] )

D = make_discriminator()
D.name = "Discr"
D_input = Input( shape=(n_features,), name='D_input' )
D_output = D(D_input)
discriminator = Model( D_input, D_output )

discriminator.name = "Discriminator"
discriminator.compile( loss='binary_crossentropy',
                            optimizer=d_optimizer,
                            metrics=['accuracy'] )
discriminator.summary()

# For the combined model we will only train the generator
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
GAN_input  = Input( shape=(GAN_noise_size,) )
GAN_latent = generator(GAN_input)
GAN_output = discriminator(GAN_latent)
GAN        = Model( GAN_input, GAN_output )
GAN.name   = "GAN"
GAN.compile( loss='binary_crossentropy',
                  optimizer=g_optimizer )
GAN.summary()


print "INFO: saving models to png files"
plot_model( generator,      to_file="img/model_%s_generator.png" % (dsid) )
plot_model( discriminator,  to_file="img/model_%s_discriminator.png" % (dsid) )
plot_model( GAN,            to_file="img/model_%s_GAN.png" % (dsid) )

# Training:
# 1) pick up ntrain events from real dataset
# 2) generate ntrain fake events

# Pre-train discriminator
#ntrain = 20000
#train_idx = random.sample( range(0,X_train.shape[0]), ntrain)
#X_train_real = X_train[train_idx,:]

#X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_noise_size])
#X_noise = np.random.uniform(-1,1,size=[X_train_real.shape[0], GAN_noise_size])
#X_noise = np.random.normal( 0., 1., (X_train_real.shape[0], GAN_noise_size) )
#X_train_fake = generator.predict(X_noise)

# create GAN training dataset
#X = np.concatenate( (X_train_real, X_train_fake) )
#n = X_train_real.shape[0]
#y = np.zeros([2*n])
#y[:n] = 1
#y[n:] = 0

#print "INFO: pre-training discriminator network"
#discriminator.trainable = True
#for layer in discriminator.layers:
#    layer.trainable = True
##discriminator.fit( X, [y,y], epochs=1, batch_size=128)
#discriminator.fit( X, y, epochs=1, batch_size=128)
###y_hat = discriminator.predict(X)

history = {
   "d_loss_orig":[], "d_loss_r_orig":[], "d_loss_f_orig":[],
   "g_loss_orig":[],
   "d_acc_orig":[], "d_acc_r_orig":[], "d_acc_f_orig":[],

   "d_loss_flip":[], "d_loss_r_flip":[], "d_loss_f_flip":[],
   "g_loss_flip":[],
   "d_acc_flip":[], "d_acc_r_flip":[], "d_acc_f_flip":[],

   "g_loss_mean" : [], "d_loss_mean" : [], "d_acc_mean":[],
}

#######################

def train_loop(nb_epoch=1000, BATCH_SIZE=32 ):

   plt_frq = max( 1, int(nb_epoch)/20 )

   #lr = float( K.get_value( discriminator.optimizer.lr ) )

   y_real = np.ones(  (BATCH_SIZE,1) )
   y_fake = np.zeros( (BATCH_SIZE,1) )

   for epoch in range(nb_epoch):

        # select some real events
        train_idx = np.random.randint( 0, X_train.shape[0], size=BATCH_SIZE )
        X_train_real = X_train[train_idx,:]

        # generate fake events
        #X_noise = np.random.uniform(-1,1,size=[X_train_real.shape[0], GAN_noise_size])
        X_noise = np.random.uniform(0,1,size=[BATCH_SIZE, GAN_noise_size])
        #X_noise = np.random.normal( 0., 0.5, (BATCH_SIZE, GAN_noise_size) )
        X_train_fake = generator.predict(X_noise)

        # Train the discriminator (real classified as ones and generated as zeros)
#        d_loss_orig, d_loss_r_orig, d_loss_r_flip, d_acc_r_orig, d_acc_r_flip = discriminator.train_on_batch( [X_train_real], [ y_real, y_real ] )
#        d_loss_flip, d_loss_f_orig, d_loss_f_flip, d_acc_f_orig, d_acc_f_flip = discriminator.train_on_batch( [X_train_fake], [ y_fake, y_fake ] )
        d_loss_r, d_acc_r = discriminator.train_on_batch( X_train_real, y_real )
        d_loss_f, d_acc_f = discriminator.train_on_batch( X_train_fake, y_fake )

        #hack
        d_loss_r_orig = d_loss_r
        d_loss_r_flip = d_loss_r
        d_loss_f_orig = d_loss_f
        d_loss_f_flip = d_loss_f
        d_loss_orig = 0.5 * np.add(d_loss_r_orig, d_loss_f_orig)
        d_acc_r_orig = d_acc_r
        d_acc_r_flip = d_acc_r
        d_acc_f_orig = d_acc_f
        d_acc_f_flip = d_acc_f

        d_loss_orig = d_loss_r + d_loss_f
        d_loss_flip = d_loss_r + d_loss_f
        #/hack

        d_loss_orig /= 2.
        d_loss_flip /= 2.
        d_acc_orig  = 0.5 * np.add(d_acc_r_orig, d_acc_f_orig)
        d_acc_flip  = 0.5 * np.add(d_acc_r_flip, d_acc_f_flip)
        d_loss_mean = 0.25 * ( d_loss_r_orig + d_loss_f_orig + d_loss_r_flip + d_loss_f_flip )
        d_acc_mean  = 0.25 * ( d_acc_r_orig + d_acc_f_orig + d_acc_r_flip + d_acc_f_flip )

        history["d_loss_orig"].append(d_loss_orig)
        history["d_loss_r_orig"].append(d_loss_r_orig)
        history["d_loss_f_orig"].append(d_loss_f_orig)
        history["d_acc_orig"].append(d_acc_orig)
        history["d_acc_f_orig"].append(d_acc_f_orig)
        history["d_acc_r_orig"].append(d_acc_r_orig)

        history["d_loss_flip"].append(d_loss_flip)
        history["d_loss_r_flip"].append(d_loss_r_flip)
        history["d_loss_f_flip"].append(d_loss_f_flip)
        history["d_acc_flip"].append(d_acc_flip)
        history["d_acc_f_flip"].append(d_acc_f_flip)
        history["d_acc_r_flip"].append(d_acc_r_flip)

        history["d_loss_mean"].append( d_loss_mean )
        history["d_acc_mean"].append( d_acc_mean )

        # Train the generator
        # create new (statistically independent) random noise sample
        #X_noise = np.random.uniform(-1,1,size=(BATCH_SIZE, GAN_noise_size))
        #X_noise = np.random.uniform(0,1,size=(BATCH_SIZE, GAN_noise_size))
        #X_noise = np.random.normal( 0., 1., (BATCH_SIZE, GAN_noise_size) )

        # we want discriminator to mistake images as real
#        g_loss_mean, g_loss_orig, g_loss_flip = GAN.train_on_batch( X_noise, [ y_real, y_real ] )
#        g_loss_mean /= 2.
        g_loss_mean = GAN.train_on_batch( X_noise, y_real )
        g_loss_orig = g_loss_mean
        g_loss_flip = g_loss_mean
        history["g_loss_orig"].append(g_loss_orig)
        history["g_loss_flip"].append(g_loss_flip)
        history["g_loss_mean"].append(g_loss_mean)

        if epoch % plt_frq == 0:
           print "Epoch: %5i :: BS = %i :: d_loss_orig = %.2f ( real = %.2f, fake = %.2f ), d_acc_orig = %.2f ( real = %.2f, fake = %.2f ), g_loss_orig = %.2f" % (
              epoch, BATCH_SIZE, d_loss_orig, d_loss_r_orig, d_loss_f_orig, d_acc_orig, d_acc_r_orig, d_acc_f_orig, g_loss_orig )
           print "Epoch: %5i :: BS = %i :: d_loss_flip = %.2f ( real = %.2f, fake = %.2f ), d_acc_flip = %.2f ( real = %.2f, fake = %.2f ), g_loss_flip = %.2f" % (
              epoch, BATCH_SIZE, d_loss_flip, d_loss_r_flip, d_loss_f_flip, d_acc_flip, d_acc_r_flip, d_acc_f_flip, g_loss_flip )
           print "Epoch: %5i :: d_loss_mean = %.2f, d_acc_mean = %.2f, g_loss_mean = %.2f" % ( epoch, d_loss_mean, d_acc_mean, g_loss_mean )
           print "----"

        #BATCH_SIZE = int( BATCH_SIZE / lr )

   return history
 
#######################

print "INFO: Train for %i epochs" % ( n_epochs )
train_loop( nb_epoch=n_epochs, BATCH_SIZE=32 )
#train_loop( nb_epoch=n_epochs, BATCH_SIZE=128 )
#train_loop( nb_epoch=n_epochs, BATCH_SIZE=1024 )

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

h_d_loss_orig   = TGraphErrors()
h_d_loss_r_orig = TGraphErrors()
h_d_loss_f_orig = TGraphErrors()
h_d_acc_orig    = TGraphErrors()
h_g_loss_orig   = TGraphErrors()
h_d_acc_orig    = TGraphErrors()
h_d_acc_f_orig  = TGraphErrors()
h_d_acc_r_orig  = TGraphErrors()

h_d_loss_flip   = TGraphErrors()
h_d_loss_r_flip = TGraphErrors()
h_d_loss_f_flip = TGraphErrors()
h_d_acc_flip    = TGraphErrors()
h_g_loss_flip   = TGraphErrors()
h_d_acc_flip    = TGraphErrors()
h_d_acc_f_flip  = TGraphErrors()
h_d_acc_r_flip  = TGraphErrors()

h_g_loss_mean = TGraphErrors()
h_d_loss_mean = TGraphErrors()
h_d_acc_mean  = TGraphErrors()

n_epochs = len(history['d_loss_orig'])
for i in range( n_epochs ):
      d_loss_orig   = history['d_loss_orig'][i]
      d_loss_r_orig = history['d_loss_r_orig'][i]
      d_loss_f_orig = history['d_loss_f_orig'][i]
      d_acc_orig    = history['d_acc_orig'][i]
      d_acc_f_orig  = history['d_acc_f_orig'][i]
      d_acc_r_orig  = history['d_acc_r_orig'][i]
      g_loss_orig   = history['g_loss_orig'][i]
      
      h_d_loss_orig.SetPoint( i, i, d_loss_orig )
      h_d_loss_r_orig.SetPoint( i, i, d_loss_r_orig )
      h_d_loss_f_orig.SetPoint( i, i, d_loss_f_orig )
      h_d_acc_orig.SetPoint( i, i, d_acc_orig )
      h_d_acc_f_orig.SetPoint( i, i, d_acc_f_orig )
      h_d_acc_r_orig.SetPoint( i, i, d_acc_r_orig )
      h_g_loss_orig.SetPoint( i, i, g_loss_orig )

      d_loss_flip   = history['d_loss_flip'][i]
      d_loss_r_flip = history['d_loss_r_flip'][i]
      d_loss_f_flip = history['d_loss_f_flip'][i]
      d_acc_flip    = history['d_acc_flip'][i]
      d_acc_f_flip  = history['d_acc_f_flip'][i]
      d_acc_r_flip  = history['d_acc_r_flip'][i]
      g_loss_flip   = history['g_loss_flip'][i]
      
      h_d_loss_flip.SetPoint( i, i, d_loss_flip )
      h_d_loss_r_flip.SetPoint( i, i, d_loss_r_flip )
      h_d_loss_f_flip.SetPoint( i, i, d_loss_f_flip )
      h_d_acc_flip.SetPoint( i, i, d_acc_flip )
      h_d_acc_f_flip.SetPoint( i, i, d_acc_f_flip )
      h_d_acc_r_flip.SetPoint( i, i, d_acc_r_flip )
      h_g_loss_flip.SetPoint( i, i, g_loss_flip )

      g_loss_mean = history["g_loss_mean"][i]
      d_loss_mean = history["d_loss_mean"][i]
      d_acc_mean    = history['d_acc_mean'][i]
      h_g_loss_mean.SetPoint( i, i, g_loss_mean )
      h_d_loss_mean.SetPoint( i, i, d_loss_mean )
      h_d_acc_mean.SetPoint( i, i, d_acc_mean )
      
h_d_loss_orig.Write( "d_loss_orig" )
h_d_loss_r_orig.Write( "d_loss_r_orig" )
h_d_loss_f_orig.Write( "d_loss_f_orig" )
h_g_loss_orig.Write( "g_loss_orig")
h_d_acc_orig.Write( "d_acc_orig" )
h_d_acc_f_orig.Write( "d_acc_f_orig" )
h_d_acc_r_orig.Write( "d_acc_r_orig" )

h_d_loss_flip.Write( "d_loss_flip" )
h_d_loss_r_flip.Write( "d_loss_r_flip" )
h_d_loss_f_flip.Write( "d_loss_f_flip" )
h_g_loss_flip.Write( "g_loss_flip")
h_d_acc_flip.Write( "d_acc_flip" )
h_d_acc_f_flip.Write( "d_acc_f_flip" )
h_d_acc_r_flip.Write( "d_acc_r_flip" )

h_g_loss_mean.Write( "g_loss_mean" )
h_d_loss_mean.Write( "d_loss_mean" )
h_d_acc_mean.Write( "d_acc_mean" )

training_root.Write()
training_root.Close()
print "INFO: training history saved to file:", training_root.GetName()
