#!/usr/bin/env python

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

try:
    import cPickle as pickle
except:
    import pickle

np.set_printoptions(precision=4, linewidth=200, suppress=True)

from features import *

# read in input file
infilename = "csv/mc16a.mg5_dijet_ht500.reco.pt250.nominal.csv"
if len(sys.argv) > 1:
    infilename = sys.argv[1]
level = infilename.split("/")[-1].split('.')[1]

if level == "ptcl":
  features = [
    "ljet1_pt", "ljet1_eta", "ljet1_M",
    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M"
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


data = pd.read_csv(infilename, delimiter=',', names=header)
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

scaler = MinMaxScaler((-1, 1))
scaler.fit(X_train)
print "INFO: data (min,max):"
print scaler.data_min_
print scaler.data_max_
print "INFO: scale factor:"
print scaler.scale_

X_train = scaler.transform(X_train)

print "INFO: X_train after standardization:"
print X_train

scaler_filename = "GAN/scaler.%s.pkl" % level
with open(scaler_filename, "wb") as file_scaler:
    pickle.dump(scaler, file_scaler)
print "INFO: scaler saved to file", scaler_filename
