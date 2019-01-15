#!/usr/bin/env python

import os, sys

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


n_train = 100000

X_train = np.zeros( [ n_train, 4 ] )
y_train = np.zeros( [ n_train, 4 ] )

v = TLorentzVector()

print "INFO: generating %i four-momenta" % n_train
for i in range(n_train):
  px = rng.Uniform( -1000., 1000. )
  py = rng.Uniform( -1000., 1000. )
  pz = rng.Uniform( -3000., 3000. )
  m  = rng.Uniform( 0, 5000. )
  E = TMath.Sqrt( px*px + py*py + pz*pz + m*m )
    
  X_train[i][0] = px
  X_train[i][1] = py
  X_train[i][2] = pz
  X_train[i][3] =  E

  v.SetPxPyPzE( X_train[i][0], X_train[i][1], X_train[i][2], X_train[i][3] )

  y_train[i][0] = v.Pt()
  y_train[i][1] = v.Eta()
  y_train[i][2] = v.Phi()
  y_train[i][3] = v.M()

print "INFO: done generating."

print "INFO: (px,py,pz,E) before transformation:"
print X_train
print "INFO: (pT,eta,phi,M) before transformation:"
print y_train

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = X_scaler.fit_transform( X_train )
y_train = y_scaler.fit_transform( y_train )

print "INFO: (px,py,pz,E) after transformation:"
print X_train
print "INFO: (pT,eta,phi,M) after transformation:"
print y_train


#~~~~~~~~~~~~~~~~~~~~

def make_model():

   input = Input( shape=(4,) )

   x = Dense(128, activation='tanh')(input)
   x = Dense(64, activation='tanh')(x)
   #x = Dense(16, activation='tanh')(x)

   output = Dense( 4)(x)

   model = Model( input, output )
   return model

#~~~~~~~~~~~~~~~~~~~~

optimizer  = Adam(0.0001)

dnn_PxPyPzE_to_PtEtaPhiM = make_model()
dnn_PxPyPzE_to_PtEtaPhiM.name = "PxPyPzE_to_PtEtaPhiM"
dnn_PxPyPzE_to_PtEtaPhiM.compile(
  loss='mean_squared_error',
  optimizer=optimizer,
  metrics=['accuracy'] )
dnn_PxPyPzE_to_PtEtaPhiM.summary()

callbacks = [ 
   EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
   #ModelCheckpoint(filepath=weights_filename, monitor='val_loss', save_best_only=True)
]

N_EPOCHS=30
BATCH_SIZE=128

history = dnn_PxPyPzE_to_PtEtaPhiM.fit( X_train, y_train,
                                        epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                                        validation_split=0.20, shuffle=True,
                                        callbacks=callbacks)
score = dnn_PxPyPzE_to_PtEtaPhiM.evaluate( X_train, y_train )
print
print "Score:", score

y_hat = dnn_PxPyPzE_to_PtEtaPhiM.predict(X_train)
y_hat = y_scaler.inverse_transform( y_hat )
X_train = X_scaler.inverse_transform( X_train )
y_train = y_scaler.inverse_transform( y_train )

for i in range(100):
  p1 = TLorentzVector()
  p2 = TLorentzVector()

  p1.SetPtEtaPhiM( y_train[i][0], y_train[i][1], y_train[i][2], y_train[i][3] )
  p2.SetPtEtaPhiM( y_hat[i][0], y_hat[i][1], y_hat[i][2], y_hat[i][3] )

  print "%-5i ) (pT,eta,phi,E:M) = (%.1f,%.2f,%.2f,%.1f:%.1f) :: (%.1f,%.2f,%.2f,%.1f:%.1f)" % (
    i,
    p1.Pt(), p1.Eta(), p1.Phi(), p1.E(), p1.M(),
    p2.Pt(), p2.Eta(), p2.Phi(), p2.E(), p2.M() )
    

model_filename = "lorentz/PxPyPzE_to_PtEtaPhiM.h5"
dnn_PxPyPzE_to_PtEtaPhiM.save( model_filename )
print "INFO: model saved to file", model_filename

scaler_filename = "lorentz_scaler.h5"
with open( scaler_filename, "wb" ) as file_scaler:
   pickle.dump( y_scaler, file_scaler )
print "INFO: scaler saved to file", scaler_filename

from ROOT import *
training_filename = "lorentz/training_history.root"
training_file = TFile.Open( training_filename, "RECREATE" )
n_epochs = len( history.history['acc'] )
h_acc      = TH1F( "acc",      "Epoch;Training Accuracy",   n_epochs, 0.5, n_epochs+0.5 )
h_loss     = TH1F( "loss",     "Epoch;Training Loss",       n_epochs, 0.5, n_epochs+0.5 )
h_val_acc  = TH1F( "val_acc",  "Epoch;Validation Accuracy", n_epochs, 0.5, n_epochs+0.5 )
h_val_loss = TH1F( "val_loss", "Epoch;Validation Loss",     n_epochs, 0.5, n_epochs+0.5 )
for i in range( n_epochs ):
      acc      = float( "%.3f" % history.history['acc'][i] )
      val_acc  = float( "%.3f" % history.history['val_acc'][i] )
      loss     = float( "%.3f" % history.history['loss'][i] )
      val_loss = float( "%.3f" % history.history['val_loss'][i] )

      h_acc.SetBinContent( i+1, acc )
      h_loss.SetBinContent( i+1, loss )
      h_val_acc.SetBinContent( i+1, val_acc )
      h_val_loss.SetBinContent( i+1, val_loss )

training_file.Write()
training_file.Close()
print "INFO: training history saved to file", training_filename

print "INFO: done."
