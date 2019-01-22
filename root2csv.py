#!/usr/bin/env python

import os, sys, argparse
import csv
from math import pow, sqrt 

from ROOT import *

from common import *
from features import *
import helper_functions
import numpy as np

gROOT.SetBatch(True)

rng = TRandom3()

def make_csv_row( eventInfo, ljets, jj ):
  row = (
     "%i" % eventInfo['runNumber'], "%i" % eventInfo['eventNumber'], "%.3f" % eventInfo['weight'],

     # leading jet
     "%4.1f" % ljets[0].Px(),  "%4.1f" % ljets[0].Py(), "%4.1f" % ljets[0].Pz(), "%4.1f" % ljets[0].Pt(),
     "%.2f"  % ljets[0].Eta(), "%.2f"  % ljets[0].Phi(),
     "%4.1f" % ljets[0].E(),   "%4.1f" % ljets[0].M(),
     "%.3f"  % ljets[0].tau2,  "%.3f"  % ljets[0].tau3, "%.3f" % ljets[0].tau32, 

     # subleading jet
     "%4.1f" % ljets[1].Px(),  "%4.1f" % ljets[1].Py(), "%4.1f" % ljets[1].Pz(), "%4.1f" % ljets[1].Pt(),
     "%.2f"  % ljets[1].Eta(), "%.2f"  % ljets[1].Phi(),
     "%4.1f" % ljets[1].E(),   "%4.1f" % ljets[1].M(),
     "%.3f"  % ljets[1].tau2,  "%.3f"  % ljets[1].tau3, "%.3f" % ljets[1].tau32,

     # dijet system
     "%4.1f" % jj.Px(),  "%4.1f" % jj.Py(), "%4.1f" % jj.Pz(), "%4.1f" % jj.Pt(),
     "%.2f"  % jj.Eta(), "%.2f"  % jj.Phi(),
     "%4.1f" % jj.E(),   "%4.1f" % jj.M(),
     "%.2f"  % jj.dPhi,  "%.2f"  % jj.dEta, "%.2f" % jj.dR,

  )
  return row

###############################

parser = argparse.ArgumentParser(description='root to csv converter')
parser.add_argument( '-i', '--filelistname', default="filelists/data.txt" )
parser.add_argument( '-o', '--outfilename',  default="" )
parser.add_argument( '-c', '--classifier',   default="rnn:GAN" )
parser.add_argument( '-s', '--systematic',   default="nominal" )
parser.add_argument( '-p', '--preselection', default='incl' )
parser.add_argument( '-f', '--tranining_fraction', default=1.0 )

args            = parser.parse_args()
filelistname    = args.filelistname
outfilename     = args.outfilename
classifier      = args.classifier
classifier_arch, classifier_feat = classifier.split(':')
syst            = args.systematic
preselection    = args.preselection
training_fraction = abs(float(args.tranining_fraction))
if training_fraction > 1: training_fraction = 1.0

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
print "INFO: training fraction:  ", training_fraction
print "INFO: output file:        ", outfilename

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

   w = 1.
   if not dsid == 0:
#      w *= tree.weight_mc
     w = helper_functions.GetEventWeight( tree, syst )

   mcChannelNumber = tree.mcChannelNumber
   runNumber       = tree.runNumber
   eventNumber     = tree.eventNumber

   ljets = helper_functions.MakeEventJets( tree )
   ljets_n = len( ljets )

   # apply training/testing filter
   u = 0.
   if training_fraction < 1.0:
     u = rng.Uniform( 0, 1 ) 
     if u > tranining_fraction: continue

   # sort by b-tagging weight 
   # jets.sort( key=lambda jet: jet.mv2c10, reverse=True )

   lj1_phi = ljets[0].Phi()
   helper_functions.RotateJets( ljets, -lj1_phi )

   #lj1_eta = ljets[0].Eta()
   #if lj1_eta < 0:

   #for do_flip_eta in [ False, True ]:
   #  if do_flip_eta == True:

   #if ljets[0].Eta() < 0:
   #    helper_functions.FlipEta( ljets )

   for do_flip_eta in [ False, True ]:

     jj      = ljets[0] + ljets[1]
     jj.dPhi = ljets[0].DeltaPhi( ljets[1] )
     jj.dEta = ljets[0].Eta() - ljets[1].Eta()
     jj.dR   = TMath.Sqrt( jj.dPhi*jj.dPhi + jj.dEta*jj.dEta )

     eventInfo = {
       'runNumber'   : tree.runNumber,
       'eventNumber' : tree.eventNumber,
       'weight'      : w
     }
   
     csv_row = make_csv_row( eventInfo, ljets, jj )
     csvwriter.writerow( csv_row )
   
   n_good += 1

outfile.close()

f_good = 100. * n_good / n_entries
print "INFO: %i entries written (%.2f %%)" % ( n_good, f_good) 

