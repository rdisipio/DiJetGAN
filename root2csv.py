#!/usr/bin/env python

import os
import sys
import argparse
import csv
from math import pow, sqrt

from ROOT import *

from common import *
from features import *
import helper_functions
import numpy as np

gROOT.SetBatch(True)

rng = TRandom3()


def make_csv_row(eventInfo, ljets, jj):
    row = (
        "%i" % eventInfo['eventNumber'], "%.3f" % eventInfo['weight'],

        # leading jet
        "%4.1f" % ljets[0].Pt(),  "%.2f" % ljets[0].Eta(
        ), "%.2f" % ljets[0].Phi(),
        "%4.1f" % ljets[0].E(),   "%4.1f" % ljets[0].M(),

        # subleading jet
        "%4.1f" % ljets[1].Pt(),  "%.2f" % ljets[1].Eta(
        ), "%.2f" % ljets[1].Phi(),
        "%4.1f" % ljets[1].E(),   "%4.1f" % ljets[1].M(),

        # dijet system
        "%4.1f" % jj.Pt(),  "%.2f" % jj.Eta(), "%.2f" % jj.Phi(),
        "%4.1f" % jj.E(),   "%4.1f" % jj.M(),

        "%.2f" % jj.dEta,  "%.2f" % jj.dPhi, "%.2f" % jj.dR,

    )
    return row

###############################


parser = argparse.ArgumentParser(description='root to csv converter')
parser.add_argument('-i', '--filelistname', default="filelists/data.txt")
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

if outfilename == "":
    fpath = filelistname.split("/")[-1]
    if "mc16" in fpath:
        camp = fpath.split('.')[0]
        dsid = fpath.split('.')[1]
        outfilename = "csv/%s.%s.%s.%s.%s.csv" % (
            camp, dsid, level, preselection, syst)
    else:
        dsid = fpath.split('.')[0]
        outfilename = "csv/%s.%s.%s.%s.csv" % (dsid, level, preselection, syst)

print "INFO: preselection:       ", preselection
print "INFO: level:              ", level
print "INFO: training fraction:  ", training_fraction
print "INFO: output file:        ", outfilename

outfile = open(outfilename, "wt")
csvwriter = csv.writer(outfile)

treename = "nominal"
if syst in systematics_tree:
    treename = syst
else:
    treename = "nominal"

print "INFO: reading systematic", syst, "from tree", treename
tree = TChain(treename, treename)
f = open(filelistname, 'r')
for fname in f.readlines():
    fname = fname.strip()
#   print "DEBUG: adding file:", fname
    tree.AddFile(fname)

n_entries = tree.GetEntries()
print "INFO: entries found:", n_entries

n_good = 0
for ientry in range(n_entries):

    if (n_entries < 10) or ((ientry+1) % int(float(n_entries)/10.) == 0):
        perc = 100. * ientry / float(n_entries)
        print "INFO: Event %-9i  (%3.0f %%)" % (ientry, perc)

    # apply training/testing filter
    u = 0.
    if training_fraction < 1.0:
        u = rng.Uniform(0, 1)
        if u > tranining_fraction:
            continue

    tree.GetEntry(ientry)

    eventNumber = tree.eventNumber
    w = tree.weight_mc

    ljets = [TLorentzVector(), TLorentzVector()]
    lj1 = ljets[0]
    lj2 = ljets[1]

    lj1.SetPtEtaPhiM(tree.ljet1_pt,
                     tree.ljet1_eta,
                     tree.ljet1_phi,
                     tree.ljet1_m)
    lj2.SetPtEtaPhiM(tree.ljet2_pt,
                     tree.ljet2_eta,
                     tree.ljet2_phi,
                     tree.ljet2_m)

    lj1_phi = lj1.Phi()
    helper_functions.RotateJets(ljets, -lj1_phi)

    #lj1_eta = ljets[0].Eta()
    # if lj1_eta < 0:

    # for do_flip_eta in [ False, True ]:
    #  if do_flip_eta == True:

    for do_flip_eta in [False, True]:

        if do_flip_eta == True:
            helper_functions.FlipEta(ljets)

        jj = ljets[0] + ljets[1]
        jj.dPhi = ljets[0].DeltaPhi(ljets[1])
        jj.dEta = ljets[0].Eta() - ljets[1].Eta()
        jj.dR = TMath.Sqrt(jj.dPhi*jj.dPhi + jj.dEta*jj.dEta)

        eventInfo = {
            #'runNumber': tree.runNumber,
            'eventNumber': tree.eventNumber,
            'weight': w
        }

        csv_row = make_csv_row(eventInfo, ljets, jj)
        csvwriter.writerow(csv_row)

    n_good += 1

outfile.close()

f_good = 100. * n_good / n_entries
print "INFO: %i entries written (%.2f %%)" % (n_good, f_good)
