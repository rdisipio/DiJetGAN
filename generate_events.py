#!/usr/bin/env python

import os
import sys
import argparse

from ROOT import *
from array import array

import cPickle as pickle
from keras.models import load_model
import numpy as np

#import helper_functions as hf
#from fourmomentum_scaler import *
from models import *

rng = TRandom3()

parser = argparse.ArgumentParser(description="GAN event generator")
parser.add_argument('-l', '--level',             default="reco")
parser.add_argument('-p', '--preselection',      default="pt250")
parser.add_argument('-s', '--systematic',        default="nominal")
parser.add_argument('-d', '--dsid',              default="mg5_dijet_ht500")
parser.add_argument('-n', '--nevents',           default=10000)
parser.add_argument('-e', '--epoch',             default=-1)
args = parser.parse_args()

level = args.level
preselection = args.preselection
systematic = args.systematic
dsid = args.dsid
syst = args.systematic
n_events = int(args.nevents)
epoch = int(args.epoch)

ljets_n_max = 2

##################
# Load Keras stuff

dnn = None
print "INFO: Systematic:", syst
print "INFO: Level", level

#model_filename = "GAN/DCGAN.generator.%s.%s.%s.%s.epoch_95000.h5" % (
if epoch == -1:
  model_filename = "GAN_%s/DCGAN.generator.%s.%s.%s.%s.h5" % (
    level, dsid, level, preselection, systematic)
else:
  model_filename = "GAN_%s/DCGAN.generator.%s.%s.%s.%s.epoch_%05i.h5" % (
    level, dsid, level, preselection, systematic, epoch)

print "INFO: loading generator model from", model_filename
generator = load_model(model_filename, custom_objects={
                       'wasserstein_loss': wasserstein_loss})
print generator.summary()

scaler_filename = "GAN_%s/scaler.%s.pkl" % (level,level)
print "INFO: loading scaler from", scaler_filename
with open(scaler_filename, "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

GAN_noise_size = generator.layers[0].input_shape[1]
GAN_output_size = generator.layers[-1].output_shape[1]
print "GAN noise size:", GAN_noise_size
print "GAN output size:", GAN_output_size

outfname = "ntuples_GAN/tree.%s.%s.%s.%s.root" % (
    dsid, level, preselection, systematic)
outfile = TFile.Open(outfname, "RECREATE")

b_eventNumber = array('l', [0])
b_weight_mc = array('f', [1.])
b_mu = array('l', [1] )
b_ljet1_pt = array('f', [0.])
b_ljet1_eta = array('f', [0.])
b_ljet1_phi = array('f', [0.])
b_ljet1_E = array('f', [0.])
b_ljet1_m = array('f', [0.])
b_ljet2_pt = array('f', [0.])
b_ljet2_eta = array('f', [0.])
b_ljet2_phi = array('f', [0.])
b_ljet2_E = array('f', [0.])
b_ljet2_m = array('f', [0.])
b_jj_pt = array('f', [0.])
b_jj_eta = array('f', [0.])
b_jj_phi = array('f', [0.])
b_jj_E = array('f', [0.])
b_jj_m = array('f', [0.])
b_jj_dEta = array('f', [0.])
b_jj_dPhi = array('f', [0.])
b_jj_dR = array('f', [0.])

outtree = TTree(systematic, "GAN generated events")
outtree.Branch('eventNumber',        b_eventNumber,     'eventNumber/l')
outtree.Branch('weight_mc',          b_weight_mc,       'weight_mc/F')
outtree.Branch('mu',         b_mu,        'mu/l' )
outtree.Branch('ljet1_pt',   b_ljet1_pt,  'ljet1_pt/F')
outtree.Branch('ljet1_eta',  b_ljet1_eta, 'ljet1_eta/F')
outtree.Branch('ljet1_phi',  b_ljet1_phi, 'ljet1_phi/F')
outtree.Branch('ljet1_E',    b_ljet1_E,   'ljet1_E/F')
outtree.Branch('ljet1_m',    b_ljet1_m,   'ljet1_m/F')
outtree.Branch('ljet2_pt',   b_ljet2_pt,  'ljet2_pt/F')
outtree.Branch('ljet2_eta',  b_ljet2_eta, 'ljet2_eta/F')
outtree.Branch('ljet2_phi',  b_ljet2_phi, 'ljet2_phi/F')
outtree.Branch('ljet2_E',    b_ljet2_E,   'ljet2_E/F')
outtree.Branch('ljet2_m',    b_ljet2_m,   'ljet2_m/F')
outtree.Branch('jj_pt',   b_jj_pt,  'jj_pt/F')
outtree.Branch('jj_eta',  b_jj_eta, 'jj_eta/F')
outtree.Branch('jj_phi',  b_jj_phi, 'jj_phi/F')
outtree.Branch('jj_E',    b_jj_E,   'jj_E/F')
outtree.Branch('jj_m',    b_jj_m,   'jj_m/F')
outtree.Branch('jj_dEta', b_jj_dEta, 'jj_dEta/F')
outtree.Branch('jj_dPhi', b_jj_dPhi, 'jj_dPhi/F')
outtree.Branch('jj_dR',   b_jj_dR,   'jj_dR/F')

print "INFO: generating %i events..." % n_events

X_noise = np.random.uniform(0, 1, size=[n_events, GAN_noise_size])
#X_noise = np.random.uniform(-1,1,size=[ n_events, GAN_noise_size])
#X_noise = np.random.normal( 0., 1, size=[ n_events, GAN_noise_size] )
X_generated = generator.predict(X_noise)

print "INFO: generated %i events" % n_events

X_generated = scaler.inverse_transform(X_generated)

print "INFO: ...done."
print

print "INFO: filling tree..."
n_good = 0
for ievent in range(n_events):
    if (n_events < 10) or ((ievent+1) % int(float(n_events)/10.) == 0):
        perc = 100. * ievent / float(n_events)
        print "INFO: Event %-9i  (%3.0f %%)" % (ievent, perc)

    # event weight
    w = 1.0

    b_eventNumber[0] = ievent
    b_weight_mc[0] = 1.0
    b_mu[0] = 1

    ljets = [TLorentzVector(), TLorentzVector()]
    ljet1 = ljets[0]
    ljet1.SetPtEtaPhiM(X_generated[ievent][0],
                       X_generated[ievent][1],
                       0.,
                       X_generated[ievent][2])

    ljet2 = ljets[1]
    ljet2.SetPtEtaPhiM(X_generated[ievent][3],
                       X_generated[ievent][4],
                       X_generated[ievent][5],
                       X_generated[ievent][6])
    if level == "reco":
       b_mu[0] = int( X_generated[ievent][7] )

    # rotate jets' P4's to make flat distribution
    if rng.Uniform() > 0.5:
        ljet2.SetPtEtaPhiM(ljet2.Pt(), ljet2.Eta(), -ljet2.Phi(), ljet2.M())
    phi = rng.Uniform(-TMath.Pi(), TMath.Pi())
    ljet1.RotateZ(phi)
    ljet2.RotateZ(phi)

    # flip eta?
    if rng.Uniform() > 0.5:
        ljet1.SetPtEtaPhiM(ljet1.Pt(), -ljet1.Eta(), ljet1.Phi(), ljet1.M())
        ljet2.SetPtEtaPhiM(ljet2.Pt(), -ljet2.Eta(), ljet2.Phi(), ljet2.M())

    if ljet1.Pt() < ljet2.Pt():
        continue
    if ljet1.Pt() < 250:
        continue
    if ljet2.Pt() < 250:
        continue
    # if abs(lj1.Eta()) > 2.0:
    #    continue
    # if abs(lj2.Eta()) > 2.0:
    #    continue

    n_good += 1

    jj = ljet1 + ljet2
    jj.dEta = ljet1.Eta() - ljet2.Eta()
    jj.dPhi = ljet1.DeltaPhi(ljet2)
    jj.dR = ljet1.DeltaR(ljet2)

    # Fill branches
    b_ljet1_pt[0] = ljet1.Pt()
    b_ljet1_eta[0] = ljet1.Eta()
    b_ljet1_phi[0] = ljet1.Phi()
    b_ljet1_E[0] = ljet1.E()
    b_ljet1_m[0] = ljet1.M()

    b_ljet2_pt[0] = ljet2.Pt()
    b_ljet2_eta[0] = ljet2.Eta()
    b_ljet2_phi[0] = ljet2.Phi()
    b_ljet2_E[0] = ljet2.E()
    b_ljet2_m[0] = ljet2.M()

    b_jj_pt[0] = jj.Pt()
    b_jj_eta[0] = jj.Eta()
    b_jj_phi[0] = jj.Phi()
    b_jj_E[0] = jj.E()
    b_jj_m[0] = jj.M()

    outtree.Fill()

    # end event loop

outtree.Write()
outfile.Close()

f_good = 100. * float(n_good) / float(n_events)
print "INFO: saved %i events (%i%%)" % (n_good, f_good)

print "INFO: output file created:", outfname
print "INFO: done."
