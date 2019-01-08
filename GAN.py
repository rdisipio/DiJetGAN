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

from models import *

############

def GenerateEvents( n_events=1000 ):

   X_noise = np.random.uniform(0,1,size=[ n_events, GAN_input_size])
   X_generated = generator.predict(X_noise)

   print "INFO: generated %i events" % n_events

   X_generated = scaler.inverse_transform( X_generated )

   X_generated = X_generated.reshape( (n_events, n_fso_max, n_features_per_fso) )
   print "INFO: reshaped:", (n_events, n_fso_max, n_features_per_fso)
   
   return X_generated   

############


known_classifiers = [ "rnn:GAN", "rnn:highlevel", "rnn:PxPyPzMBwNtrk" ]

parser = argparse.ArgumentParser(description='ttbar diffxs sqrt(s) = 13 TeV classifier training')
parser.add_argument( '-i', '--training_filename', default="" )
parser.add_argument( '-c', '--classifier',        default=known_classifiers[0] )
parser.add_argument( '-p', '--preselection',      default="2b_incl" )
parser.add_argument( '-s', '--systematic',        default="nominal" ) 
parser.add_argument( '-g', '--gpus_n',            default=1, type=int )
parser.add_argument( '-d', '--dsid',              default="410471" )
parser.add_argument( '-n', '--nepochs',           default=1000 )
args         = parser.parse_args()

classifier        = args.classifier
training_filename = args.training_filename
preselection      = args.preselection
systematic        = args.systematic
gpus_n            = args.gpus_n
dsid              = args.dsid
n_epochs          = int(args.nepochs)

classifier_arch, classifier_feat = classifier.split(':')

if training_filename == "":
#   training_filename = "csv/training.%s.%s.%s.%s.csv" % ( classifier_arch, classifier_feat, preselection, systematic )
#   training_filename = "csv/mc16a.361025.%s.%s.%s.%s.csv" % ( classifier_arch, classifier_feat, preselection, systematic )
   training_filename = "csv/mc16a.%s.%s.%s.%s.%s.csv" % ( dsid, classifier_arch, classifier_feat, preselection, systematic )
   print "INFO: training file:", training_filename
else:
   systematic = training_filename.split("/")[-1].split('.')[-2]
   
print "INFO: training systematic: %s" % systematic

#scaler = StandardScaler()
scaler = MinMaxScaler( (-1,1) )

from features_GAN import *

#features_GAN = [
#   "ljet1_px", "ljet1_py", "ljet1_pz", "ljet1_M", #"ljet1_tau32",
#   "ljet2_px", "ljet2_py", "ljet2_pz", "ljet2_M", #"ljet2_tau32",
#]

features_GAN = [
    "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",# "ljet1_tau32",
    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",# "ljet2_tau32",
   ]

header     = header_GAN
features   = features_GAN
n_features = len(features) 
n_features_per_fso = int( n_features / n_fso_max )
print "INFO: total number of input features:     ", n_features
print "INFO: max number of objects:              ", n_fso_max 
print "INFO: number of input features per object:", n_features_per_fso

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

print "INFO: input features:"
print features
X_train = data[features].values

print "INFO: X_train before standardization:"
print X_train

X_train = scaler.fit_transform( X_train )
print "INFO: X_train after standardization:"
print X_train

n_events = len( X_train )
print "INFO: number of training events:", n_events

#if classifier_arch == "rnn":
#   print "INFO: RNN requires reshaping:", (n_fso_max, n_features_per_fso)
#   X_train = X_train.reshape( (n_events, n_fso_max, n_features_per_fso) )

print "INFO: X_train shape:", X_train.shape
print X_train

# Use MC event weights for training?
event_weights = None
#event_weights = data["weight"].values
print "INFO: event weights:"
print event_weights

GAN_input_size = 64 # number of random numbers (input noise)

#~~~~~~~~~~~~~~~~~~~~~~

def make_generator_mlp():
   # Build Generative model ...

   G_input = Input( shape=(GAN_input_size,) )

   G = Dense( 64, kernel_initializer='glorot_normal' )(G_input)
   G = Activation('tanh')(G)
   G = BatchNormalization(momentum=0.8)(G) #0.8

   G = Dense( 32 )(G)
   G = Activation('tanh')(G)
   G = BatchNormalization(momentum=0.8)(G) #0.8

   G = Dense( 16 )(G)
   G = Activation('tanh')(G)

   G = Dense( n_features, activation="tanh" )(G)

   generator = Model( G_input, G )

   return generator

#~~~~~~~~~~~~~~~~~~~~~~

def make_discriminator_mlp():
   # Build Discriminative model ...
   inshape = ( n_features, )
   D_input = Input( shape=inshape, name='D_input' )

   D = Dense( 64 )(D_input)
   D = Activation('tanh')(D)
   D = BatchNormalization(momentum=0.99)(D) # 0.8

   D = Dense( 32 )(D)
   D = Activation('tanh')(D)
   D = BatchNormalization(momentum=0.99)(D)
   
   D = Dense( 16 )(D)
   D = Activation('tanh')(D)
   #D = BatchNormalization(momentum=0.99)(D)

   #D = Dense( 8 )(D)
   #D = Activation('tanh')(D)
   #D = BatchNormalization(momentum=0.99)(D)
   
   #D = Dense( 4 )(D)
   #D = Activation('elu')(D)

   #D_output = Dense( 2, activation="softmax")(D)
   D_output = Dense( 1, activation="sigmoid")(D)
   discriminator = Model( D_input, D_output )
   #discriminator.compile( loss='categorical_crossentropy', optimizer=dopt )
   
   return discriminator

#~~~~~~~~~~~~~~~~~~~~~~

def make_generator_cnn():
   # Build Generative model ...
   
   G_input = Input( shape=(GAN_input_size,) )
   
   G = Dense( 64, kernel_initializer='glorot_uniform' )(G_input)
   G = Activation('tanh')(G)
   G = BatchNormalization(momentum=0.8)(G)
   
   G = Reshape( [ 8, 8, 1 ] )(G) #default: channel last

   G = Conv2D( filters=32, kernel_size=3, padding="same" )(G)
   G = Activation('tanh')(G)

   G = Conv2D( filters=64, kernel_size=3, padding="same" )(G)
   G = Activation('tanh')(G)

   # Upsample to make the input larger
   #G = UpSampling2D(size=2)(G)
   #G = Conv2D( filters=8, kernel_size=3, strides=1, padding='same' )(G)
   # same thing, quicker but introduces artifacts:
   #G = Conv2DTranspose( filters=4, kernel_size=4, strides=2, padding='same')(G)
   #G = Activation('tanh')(G)
   #G = BatchNormalization(momentum=0.99)(G)

   #G = Conv2D( filters=8, kernel_size=3, padding="same" )(G)
   #G = Activation('tanh')(G)
   #G = BatchNormalization(momentum=0.99)(G)
   
   #G = Conv2D( filters=16, kernel_size=2, padding="same" )(G)
   #G = Activation('tanh')(G)
   #G = BatchNormalization(momentum=0.99)(G)
   
   #G = Conv2D( filters=16, kernel_size=4, padding="same" )(G)
   #G = Activation('tanh')(G)
   #G = BatchNormalization(momentum=0.8)(G)
   
   #G = MaxPooling2D( (2,2) )(G)
   
   G = Flatten()(G)
   G = Dense( n_features, activation="tanh" )(G)
   #G = Dropout(0.2)(G)
   
   generator = Model( G_input, G )
   
   return generator

#~~~~~~~~~~~~~~~~~~~~~~

def make_discriminator_cnn():
   # Build Discriminative model ...
   inshape = ( n_features, )
   D_input = Input( shape=inshape, name='D_input' )

   #D = Reshape( (-1,n_fso_max, n_features_per_fso) )(D_input)
   D = Dense(256)(D_input)
   D = Reshape( (1,16,16) )(D)
   
   D = Conv2D( 128, 1, strides=1 )(D)
   D = Activation('tanh')(D)

   D = Conv2D( 64, 1, strides=1 )(D)
   D = Activation('tanh')(D)

   D = Flatten()(D)
   D = Dropout(0.2)(D)
   
   #D_output = Dense( 2, activation="softmax")(D)
   D_output = Dense( 1, activation="sigmoid")(D)
   discriminator = Model( D_input, D_output )
   
   return discriminator

#~~~~~~~~~~~~~~~~~~~~~~

def make_generator_rnn():

   G_input = Input( shape=(GAN_input_size,) )

   G = Dense( 128, kernel_initializer='glorot_normal' )(G_input)
   G = Activation('tanh')(G)
   G = BatchNormalization(momentum=0.99)(G) #0.8

   G = Reshape( (32,4) )(G)

   #G = Bidirectional( LSTM( 32, return_sequences=True  ) )(G)
   #G = Bidirectional( LSTM( 8, return_sequences=True ) )(G)
   G = LSTM( 32, return_sequences=True )(G)
   G = LSTM( 16, return_sequences=False )(G) #kernel_regularizer=regularizers.l2(0.01)
   G = Activation('tanh')(G)

   G = Dense( n_features, activation="tanh" )(G)

   generator = Model( G_input, G )

   return generator

#~~~~~~~~~~~~~~~~~~~~~~

def make_discriminator_rnn():

   inshape = ( n_features, )
   D_input = Input( shape=inshape, name='D_input' )
   
   D = Dense( 128, kernel_initializer='glorot_normal' )(D_input)
   D = Activation('tanh')(D)
   D = Reshape( (16,8) )(D)

   #D = Bidirectional( LSTM( 16, return_sequences=True  ) )(D)
   
   D = Bidirectional( LSTM( 8, return_sequences=False ) )(D)
   D = Activation('tanh')(D)

    #D_output = Dense( 2, activation="softmax")(D)
   D_output = Dense( 1, activation="sigmoid")(D)
   discriminator = Model( D_input, D_output )
   #discriminator.compile( loss='categorical_crossentropy', optimizer=dopt )
   
   return discriminator

#~~~~~~~~~~~~~~~~~~~~~~
   
def make_generator():
   return make_generator_mlp()
   #return make_generator_rnn()
   #return make_generator_cnn()

def make_discriminator():
   #return make_discriminator_mlp()
   #return make_discriminator_rnn()
   return make_discriminator_cnn()

d_optimizer   = Adam(0.0001) #(0.0001, 0.5)
g_optimizer   = Adam(0.0001) #(0.0001, 0.5)

discriminator = make_discriminator()
discriminator.name = "discriminator"
discriminator.compile( loss='binary_crossentropy',
                       optimizer=d_optimizer,
                       metrics=['accuracy'] )
discriminator.summary()

generator = make_generator()
generator.name = "generator"
generator.compile( loss='mean_squared_error',
                   optimizer=g_optimizer )
generator.summary()

# For the combined model we will only train the generator
discriminator.trainable = False

GAN_input  = Input( shape=(GAN_input_size,) )
GAN_hidden = generator(GAN_input)
GAN_output = discriminator(GAN_hidden)
GAN = Model( GAN_input, GAN_output )
GAN.name = "GAN"
#GAN.compile( loss='categorical_crossentropy', optimizer=opt )
GAN.compile( loss='binary_crossentropy',
             optimizer=g_optimizer )
GAN.summary()

# Training: 
# 1) pick up ntrain events from real dataset
# 2) generate ntrain fake events
ntrain = 10000

train_idx = random.sample( range(0,X_train.shape[0]), ntrain)
X_train_real = X_train[train_idx,:]

X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_input_size])
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
        X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_input_size])
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
        X_noise = np.random.uniform(0,1,size=[X_train_real.shape[0], GAN_input_size])
        X_train_fake = generator.predict(X_noise)

        # we want discriminator to mistake images as real
        g_loss = GAN.train_on_batch( X_noise, y_real )
        history["g_loss"].append(g_loss)

        if epoch % plt_frq == 0:
           print "Epoch: %5i :: d_loss = %.3f ( real = %.3f, fake = %.3f ) :: g_loss = %.3f" % ( epoch, d_loss, d_loss_real, d_loss_fake, g_loss )

   return history
 
#######################

print "INFO: Train for %i epochs at original learning rates" % ( n_epochs )
train_loop( nb_epoch=n_epochs, BATCH_SIZE=128 )

#print "INFO: train with larger batch size"
#train_loop( nb_epoch=int(n_epochs/5), BATCH_SIZE=512 )

#print "INFO: train with larger batch size"
#train_loop( nb_epoch=int(n_epochs/10), BATCH_SIZE=1024 )

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

h_d_loss = TGraphErrors()
h_d_loss_r = TGraphErrors()
h_d_loss_f = TGraphErrors()
h_d_acc  = TGraphErrors()
h_g_loss = TGraphErrors()
h_g_acc  = TGraphErrors()

h_d_loss.SetLineColor(kRed)
h_g_loss.SetLineColor(kBlue)

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

# print "INFO: saving histograms"
from common import GeV, TeV

from helper_functions import Interpolate, TopSubstructureTagger

outfname = "GAN/histograms.%s.%s.%s.%s.%s.root" % (dsid,classifier_arch, classifier_feat, preselection, systematic)  
outfile = TFile.Open( outfname, "RECREATE" )

n_events = 200000
X_generated = GenerateEvents( n_events )

# Create output tree
from array import array

b_eventNumber     = array( 'l', [ 0 ] )
b_abcd16          = array( 'i', [0] )
b_ljet_px = vector('float')(2)
b_ljet_py = vector('float')(2)
b_ljet_pz = vector('float')(2)
b_ljet_pt = vector('float')(2)
b_ljet_eta = vector('float')(2)
b_ljet_phi = vector('float')(2)
b_ljet_E   = vector('float')(2)
b_ljet_m  = vector('float')(2)
b_ljet_tau2  = vector('float')(2)
b_ljet_tau3  = vector('float')(2)
b_ljet_tau32 = vector('float')(2)
b_ljet_bmatch70_dR = vector('float')(2)
b_ljet_bmatch70    = vector('int')(2)
b_ljet_isTopTagged80  = vector('int')(2)


outtree = TTree( systematic, "GAN generated events" )
outtree.Branch( 'eventNumber',        b_eventNumber,     'eventNumber/l' )
outtree.Branch( 'ljet_px',            b_ljet_px )
outtree.Branch( 'ljet_py',            b_ljet_py )
outtree.Branch( 'ljet_pz',            b_ljet_pz )
outtree.Branch( 'ljet_pt',            b_ljet_pt )
outtree.Branch( 'ljet_eta',           b_ljet_eta )
outtree.Branch( 'ljet_phi',           b_ljet_phi )
outtree.Branch( 'ljet_E',             b_ljet_E  )
outtree.Branch( 'ljet_m',             b_ljet_m  )
outtree.Branch( 'ljet_tau2',          b_ljet_tau2 )
outtree.Branch( 'ljet_tau3',          b_ljet_tau2 )
outtree.Branch( 'ljet_tau32',         b_ljet_tau32 )
outtree.Branch( 'ljet_bmatch70_dR',   b_ljet_bmatch70_dR )
outtree.Branch( 'ljet_bmatch70',      b_ljet_bmatch70 )
#outtree.Branch( 'ljet_isTopTagged80', b_ljet_isTopTagged80 )
outtree.Branch( 'ljet_smoothedTopTaggerMassTau32_topTag80',  b_ljet_isTopTagged80 )
outtree.Branch( 'abcd16',             b_abcd16, 'abcd16/I' )

print "INFO: starting event loop:", n_events

for ievent in range(n_events):
   if ( n_events < 10 ) or ( (ievent+1) % int(float(n_events)/10.)  == 0 ):
      perc = 100. * ievent / float(n_events)
      print "INFO: Event %-9i  (%3.0f %%)" % ( ievent, perc )

   # event weight
   w = 1.0

   b_eventNumber[0] = ievent

   ljets = []
   for i in range(n_fso_max):
      ljets += [ TLorentzVector() ]
      lj = ljets[-1]
   
      pt    = X_generated[ievent][i][0]
      eta   = X_generated[ievent][i][1]
      phi   = X_generated[ievent][i][2]
      E     = max( 0., X_generated[ievent][i][3] )
      m     = max( 0., X_generated[ievent][i][4] )
      #tau32 = -1. #max( X_generated[ievent][i][4], 0. )
      lj.SetPtEtaPhiM( pt, eta, phi, m )
      
      #px    = X_generated[ievent][i][0]
      #py    = X_generated[ievent][i][1]
      #pz    = X_generated[ievent][i][2]
      #pt    = max( 0., X_generated[ievent][i][3] )
      #E     = max( 0., X_generated[ievent][i][3] )
      #m     = max( 0., X_generated[ievent][i][3] )
      #E     = TMath.Sqrt( px*px + py*py + pz*pz + m*m )
      #lj.SetPxPyPzE( px, py, pz, E )
     
      lj.tau2 = -1.
      lj.tau3 = -1.
      lj.tau32 = -1
      lj.isTopTagged80 = TopSubstructureTagger( lj )
      lj.bmatch70_dR = 10.
      lj.bmatch70    = -1

   # sort jets by pT
   ljets.sort( key=lambda jet: jet.Pt(), reverse=True )
   lj1 = ljets[0]
   lj2 = ljets[1]
   
   # Fill branches
   b_ljet_px[0]  = lj1.Px()*GeV
   b_ljet_py[0]  = lj1.Py()*GeV
   b_ljet_pz[0]  = lj1.Pz()*GeV
   b_ljet_pt[0]  = lj1.Pt()*GeV
   b_ljet_eta[0] = lj1.Eta()
   b_ljet_phi[0] = lj1.Phi()
   b_ljet_E[0]   = lj1.E()*GeV
   b_ljet_m[0]   = lj1.M()*GeV
   b_ljet_tau2[0]  = lj1.tau2
   b_ljet_tau3[0]  = lj1.tau3
   b_ljet_tau32[0] = lj1.tau32
   b_ljet_bmatch70_dR[0]   = lj1.bmatch70_dR
   b_ljet_bmatch70[0]      = lj1.bmatch70
   b_ljet_isTopTagged80[0] = lj1.isTopTagged80

   b_ljet_px[1]  = lj2.Px()*GeV
   b_ljet_py[1]  = lj2.Py()*GeV
   b_ljet_pz[1]  = lj2.Pz()*GeV
   b_ljet_pt[1]  = lj2.Pt()*GeV
   b_ljet_eta[1] = lj2.Eta()
   b_ljet_phi[1] = lj2.Phi()
   b_ljet_E[1]   = lj2.E()*GeV
   b_ljet_m[1]   = lj2.M()*GeV
   b_ljet_tau2[1]  = lj2.tau2
   b_ljet_tau3[1]  = lj2.tau3
   b_ljet_tau32[1] = lj2.tau32
   b_ljet_bmatch70_dR[1]   = lj2.bmatch70_dR
   b_ljet_bmatch70[1]      = lj2.bmatch70
   b_ljet_isTopTagged80[1] = lj2.isTopTagged80
   
   b_abcd16[0] = 0
   if lj1.isTopTagged80 == True: b_abcd16[0] ^= 0b1000
   if lj1.bmatch70 > 0:          b_abcd16[0] ^= 0b0100
   if lj2.isTopTagged80 == True: b_abcd16[0] ^= 0b0010
   if lj2.bmatch70 > 0:          b_abcd16[0] ^= 0b0001

   outtree.Fill()

outtree.Write()
outfile.Close()

print "INFO: done."

