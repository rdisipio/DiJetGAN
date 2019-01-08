#!/usr/bin/env python

import os, sys, argparse
import csv
from math import pow, sqrt 

from ROOT import *

from common import *
from features_GAN import *
import helper_functions
import numpy as np

gROOT.SetBatch(True)

rng = TRandom3()


###############################

parser = argparse.ArgumentParser(description='root to csv converter')
parser.add_argument( '-i', '--filelistname', default="filelists/data.txt" )
parser.add_argument( '-o', '--outfilename',  default="" )
parser.add_argument( '-c', '--classifier',   default="rnn:GAN" )
parser.add_argument( '-s', '--systematic',   default="nominal" )
parser.add_argument( '-p', '--preselection', default='2b_incl' )
parser.add_argument( '-a', '--data-augmentation',  default=0 )
parser.add_argument( '-f', '--tranining_fraction', default=1.0 )

args            = parser.parse_args()
filelistname    = args.filelistname
outfilename     = args.outfilename
classifier      = args.classifier
classifier_arch, classifier_feat = classifier.split(':')
syst            = args.systematic
preselection    = args.preselection
data_aug        = int( args.data_augmentation )
tranining_fraction = abs(float(args.tranining_fraction))
if tranining_fraction > 1: tranining_fraction = 1.0

if outfilename == "":
  fpath = filelistname.split("/")[-1]
  if "mc16" in fpath:
    camp = fpath.split('.')[0] 
    dsid = fpath.split('.')[1]
    outfilename = "csv/%s.%s.%s.%s.%s.%s.csv" % ( camp, dsid, classifier_arch, classifier_feat, preselection, syst )
  else:
    dsid = fpath.split('.')[0]    
    outfilename = "csv/%s.%s.%s.%s.%s.csv" % ( dsid, classifier_arch, classifier_feat, preselection, syst )

print "INFO: preselection:       ", preselection
print "INFO: classifier arch:    ", classifier_arch
print "INFO: classifier features:", classifier_feat
print "INFO: training fraction:  ", tranining_fraction
print "INFO: output file:        ", outfilename
if data_aug > 0:
   print "INFO: using data augmentation:", data_aug

outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )

treename = "nominal"
if syst in systematics_tree:
   treename = syst
else:
   treename = "nominal"

print "INFO: reading systematic", syst, "from tree", treename
tree = TChain( treename, treename )
f = open( filelistname, 'r' )
for fname in f.readlines():
   fname = fname.strip()
#   print "DEBUG: adding file:", fname
   tree.AddFile( fname )

n_entries = tree.GetEntries()
print "INFO: entries found:", n_entries
   
# switch on only useful branches
branches_active = []
branches_active += branches_eventInfo
branches_active += branches_jets
branches_active += branches_ljets

sumw = None

# Is this a MC sample?
isMC = False
if "mc16" in filelistname.lower():
   isMC = True
   print "INFO: Sample is MC"

   branches_active += branches_mc
  
   # MC16a or MC16d?
   
   if isMC:
      if "mc16a" in filelistname: sumw = sumw_mc16a
      if "mc16d" in filelistname: sumw = sumw_mc16d
      if "mc16e" in filelistname: sumw = sumw_mc16e
      if "mc16f" in filelistname: sumw = sumw_mc16f

tree.SetBranchStatus( "*", 0 )
for branch in branches_active:
   print "DEBUG: active branch", branch
   tree.SetBranchStatus( branch, 1 )

n_good = 0
for ientry in range(n_entries):
   tree.GetEntry( ientry )

   if ( n_entries < 10 ) or ( (ientry+1) % int(float(n_entries)/10.)  == 0 ):
     perc = 100. * ientry / float(n_entries)
     print "INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc )
   
   dsid = tree.mcChannelNumber

   category = -1
   if dsid == 0:
      # real data
      category = 0
      #category = ( tree.eventNumber % 2 == 0 )
   else:
      # mc signal
      category = 1

   w = 1.
   if not dsid == 0:
#      w *= tree.weight_mc
     w = helper_functions.GetEventWeight( tree, syst )

   mcChannelNumber = tree.mcChannelNumber
   runNumber       = tree.runNumber
   eventNumber     = tree.eventNumber

   jets, bjets, ljets = helper_functions.MakeEventJets( tree )
   bjets_n = len( bjets )
   ljets_n = len( ljets )

   if preselection == "1b_incl": 
      if bjets_n == 0: continue
   if preselection == "2b_incl": 
      if bjets_n < 2: continue

   # apply training/testing filter
   u = rng.Uniform( 0, 1 ) 
   if u > tranining_fraction: continue

   # sort by b-tagging weight 
   jets.sort( key=lambda jet: jet.mv2c10, reverse=True )

   phi = 0.
   for i in range(data_aug+1):

      flip_eta = True if rng.Uniform()>0.5 else False

      jets_new, bjets_new, ljets_new = helper_functions.RotateEvent( jets, bjets, ljets, phi, flip_eta=flip_eta )

      shape = ( n_fso_max, n_features_per_fso )
      event = helper_functions.make_rnn_input_GAN( ljets_new, shape, do_linearize=False )

      csvwriter.writerow( (
            "%i" % tree.runNumber, "%i" % tree.eventNumber, "%.3f" % w,

            # Leading jet
            # px                   py                     pz                     pt
            "%4.1f" % event[0][0], "%4.1f" % event[0][1], "%4.1f" % event[0][2], "%4.1f" % event[0][3],
            #eta                  phi
            "%.2f" % event[0][4], "%.2f" % event[0][5],
            #E 
            "%4.1f" % event[0][6], "%4.1f" % event[0][7],
            #tau2                 #tau3                 #tau32
            "%.3f" % event[0][8], "%.3f" % event[0][9], "%.3f" % event[0][10],  

            # Subleading jet
            # px                   py                     pz                     pt
            "%4.1f" % event[1][0], "%4.1f" % event[1][1], "%4.1f" % event[1][2], "%4.1f" % event[1][3],
            #eta                  phi
            "%.2f" % event[1][4], "%.2f" % event[1][5],
            #E 
            "%4.1f" % event[1][6], "%4.1f" % event[1][7],
            #tau2                 #tau3                 #tau32
            "%.3f" % event[1][8], "%.3f" % event[1][9], "%.3f" % event[1][10],  

            "@CATEGORY@"
      ) )     

      phi = rng.Uniform( -TMath.Pi(), TMath.Pi() )

   n_good += 1 

outfile.close()

f_good = 100. * n_good / n_entries
print "INFO: %i entries written (%.2f %%)" % ( (data_aug+1)*n_good, f_good) 

