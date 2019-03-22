#!/usr/bin/env python

import os
import sys
import argparse
import csv
from math import pow, sqrt

from ROOT import *
from array import array

from common import *
from features import *
import helper_functions
import numpy as np

gROOT.SetBatch(True)

rng = TRandom3()

###############################

parser = argparse.ArgumentParser(description='root to csv converter')
parser.add_argument('-i', '--filelistname',
                    default="filelists/delphes.mg5_dijet_ht500.MC.pt250.txt")
parser.add_argument('-o', '--outfilename',  default="")
parser.add_argument('-l', '--level',        default="reco")
parser.add_argument('-s', '--systematic',   default="nominal")
parser.add_argument('-p', '--preselection', default='pt250')
parser.add_argument('-f', '--tranining_fraction', default=1.0)

args = parser.parse_args()
filelistname = args.filelistname
outfilename = args.outfilename
level = args.level
syst = args.systematic
preselection = args.preselection
training_fraction = abs(float(args.tranining_fraction))
if training_fraction > 1:
    training_fraction = 1.0

dsid = filelistname.split("/")[-1].split('.')[1]

if outfilename == "":
    fpath = filelistname.split("/")[-1]
    if "mc16" in fpath:
        camp = fpath.split('.')[0]
        dsid = fpath.split('.')[1]
        outfilename = "ntuples_MC/tree.%s.%s.%s.%s.%s.root" % (
            camp, dsid, level, preselection, syst)
    else:
        dsid = fpath.split('.')[0]
        outfilename = "ntuples_MC/tree.%s.%s.%s.%s.root" % (
            dsid, level, preselection, syst)

print "INFO: level:              ", level
print "INFO: preselection:       ", preselection
print "INFO: training fraction:  ", training_fraction
print "INFO: output file:        ", outfilename

treename = "Delphes"

print "INFO: reading systematic", syst, "from tree", treename
tree = TChain(treename, treename)
f = open(filelistname, 'r')
for fname in f.readlines():
    fname = fname.strip()
#   print "DEBUG: adding file:", fname
    tree.AddFile(fname)

n_entries = tree.GetEntries()
print "INFO: entries found:", n_entries

# switch on only useful branches
branches_active = [
    "Event.Number", "Event.ProcessID", "Event.Weight", "Vertex_size",
    "Electron.PT", "Muon.PT",
    "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.T", "Jet.Mass",
    "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.T", "GenJet.Mass",
]
tree.SetBranchStatus("*", 0)
for branch in branches_active:
    #print "DEBUG: active branch", branch
    tree.SetBranchStatus(branch, 1)

outfile = TFile.Open(outfilename, "RECREATE")
b_eventNumber = array('l', [0])
b_weight_mc = array('f', [1.])
b_mu = array('l', [1])

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

outtree = TTree(syst, "MG5 generated events")
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

n_good = 0
ientry = 0
for ientry in range(n_entries):

    tree.GetEntry(ientry)

    if (n_entries < 10) or ((ientry+1) % int(float(n_entries)/10.) == 0):
        perc = 100. * ientry / float(n_entries)
        print "INFO: Event %-9i  (%3.0f %%)" % (ientry, perc)

    b_weight_mc[0] = tree.GetLeaf("Event.Weight").GetValue(0)
    b_mu[0] = int( tree.GetLeaf("Vertex_size").GetValue(0) )
    ljets_n = tree.GetLeaf("Jet.PT").GetLen()

    if level == "ptcl":
        ljets_n = tree.GetLeaf("GenJet.PT").GetLen()
    else:
        ljets_n = tree.GetLeaf("Jet.PT").GetLen()

    # do sanity checks first
    if ljets_n < 2:
        continue

    ljet1 = TLorentzVector()
    ljet2 = TLorentzVector()

    if level == "ptcl":
        ljet1.SetPtEtaPhiM(
            tree.GetLeaf("GenJet.PT").GetValue(0),
            tree.GetLeaf("GenJet.Eta").GetValue(0),
            tree.GetLeaf("GenJet.Phi").GetValue(0),
            tree.GetLeaf("GenJet.Mass").GetValue(0))

        ljet2.SetPtEtaPhiM(
            tree.GetLeaf("GenJet.PT").GetValue(1),
            tree.GetLeaf("GenJet.Eta").GetValue(1),
            tree.GetLeaf("GenJet.Phi").GetValue(1),
            tree.GetLeaf("GenJet.Mass").GetValue(1))
    else:
        if tree.GetLeaf("Jet.Mass").GetValue(0) < 0.: continue
        if tree.GetLeaf("Jet.Mass").GetValue(1) < 0.: continue

        ljet1.SetPtEtaPhiM(
            tree.GetLeaf("Jet.PT").GetValue(0),
            tree.GetLeaf("Jet.Eta").GetValue(0),
            tree.GetLeaf("Jet.Phi").GetValue(0),
            tree.GetLeaf("Jet.Mass").GetValue(0))

        ljet2.SetPtEtaPhiM(
            tree.GetLeaf("Jet.PT").GetValue(1),
            tree.GetLeaf("Jet.Eta").GetValue(1),
            tree.GetLeaf("Jet.Phi").GetValue(1),
            tree.GetLeaf("Jet.Mass").GetValue(1))

    # more sanity checks
    if ljet1.M() < 0.: continue
    if ljet2.M() < 0.: continue

    jj = ljet1 + ljet2

    # apply further selection cuts?
#    if jj.M() < 1500.: continue
    if "ttbar" in dsid:
       if ljet1.Pt() < 350.: continue
       if ljet2.Pt() < 350.: continue
       if ljet1.M() > 500.: continue
       if ljet2.M() > 500.: continue

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

    b_jj_dEta = ljet1.Eta() - ljet2.Eta()
    b_jj_dPhi = ljet1.DeltaPhi( ljet2 )
    b_jj_dR   = ljet1.DeltaR( ljet2 )

    outtree.Fill()
    n_good += 1

outtree.Write()
outfile.Close()

f_good = 100. * n_good / n_entries
print "INFO: %i entries written (%.2f %%)" % (n_good, f_good)
