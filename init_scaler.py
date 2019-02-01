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

features = [
    "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
    "jj_pt",    "jj_eta",    "jj_phi", "jj_E", "jj_M",
    "jj_dPhi",  "jj_dEta",   "jj_dR",
]

# features = [
#    "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_M",
#    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M",
#    "jj_pt",    "jj_eta",    "jj_phi",    "jj_M",
#]

# features = [
#    "ljet1_px", "ljet1_py", "ljet1_pz", "ljet1_E",
#    "ljet2_px", "ljet2_py", "ljet2_pz", "ljet2_E",
#    "jj_px", "jj_py", "jj_pz", "jj_E",
#]

n_features = len(features)
print "INFO: input features:"
print features
print "INFO: total number of input features:     ", n_features

# read in input file
infilename = "csv/mc16a.mg5_dijet_ht500.reco.pt250.nominal.csv"
if len(sys.argv) > 1:
    infilename = sys.argv[1]
level = infilename.split("/")[-1].split('.')[1]

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
