#!/usr/bin/env python

import os, sys, argparse
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

dsid = filelistname.split("/")[-1].split('.')[1]

if outfilename == "":
  fpath = filelistname.split("/")[-1]
  if "mc16" in fpath:
    camp = fpath.split('.')[0]
    dsid = fpath.split('.')[1]
    outfilename = "ntuples_GAN/tree.%s.%s.%s.%s.%s.%s.root" % ( camp, dsid, classifier_arch, classifier_feat, preselection, syst )
  else:
    dsid = fpath.split('.')[0]
    outfilename = "ntuples_GAN/tree.%s.%s.%s.%s.%s.root" % ( dsid, classifier_arch, classifier_feat, preselection, syst )

print "INFO: preselection:       ", preselection
print "INFO: classifier arch:    ", classifier_arch
print "INFO: classifier features:", classifier_feat
print "INFO: training fraction:  ", training_fraction
print "INFO: output file:        ", outfilename

treename = "Delphes"
#if syst in systematics_tree:
#   treename = syst
#else:
#   treename = "Delphes"

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
branches_active = [
  "Event.Number", "Event.ProcessID", "Event.Weight",
  "Electron.PT", "Muon.PT",
  "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.T", "Jet.Mass",
  ]
tree.SetBranchStatus( "*", 0 )
for branch in branches_active:
   print "DEBUG: active branch", branch
   tree.SetBranchStatus( branch, 1 )

MAX_JETS_N=2

outfile = TFile.Open( outfilename, "RECREATE" )
b_eventNumber     = array( 'l', [ 0 ] )
b_runNumber       = array( 'l', [ 0 ] )
b_mcChannelNumber = array( 'l', [0] )
b_abcd16          = array( 'i', [0] )
b_ljet_px    = vector('float')(MAX_JETS_N)
b_ljet_py    = vector('float')(MAX_JETS_N)
b_ljet_pz    = vector('float')(MAX_JETS_N)
b_ljet_pt    = vector('float')(MAX_JETS_N)
b_ljet_eta   = vector('float')(MAX_JETS_N)
b_ljet_phi   = vector('float')(MAX_JETS_N)
b_ljet_E     = vector('float')(MAX_JETS_N)
b_ljet_m     = vector('float')(MAX_JETS_N)
b_ljet_tau2  = vector('float')(MAX_JETS_N)
b_ljet_tau4  = vector('float')(MAX_JETS_N)
b_ljet_tau32 = vector('float')(MAX_JETS_N)
b_ljet_bmatch70_dR = vector('float')(MAX_JETS_N)
b_ljet_bmatch70    = vector('int')(MAX_JETS_N)
b_ljet_isTopTagged80  = vector('int')(MAX_JETS_N)

outtree = TTree( syst, "MG5 generated events" )
outtree.Branch( 'eventNumber',        b_eventNumber,     'eventNumber/l' )
outtree.Branch( 'runNumber',          b_runNumber,       'runNumber/l' )
outtree.Branch( 'mcChannelNumber',    b_mcChannelNumber, "mcChannelNumber/l" )
outtree.Branch( 'ljet_px',            b_ljet_px )
outtree.Branch( 'ljet_py',            b_ljet_py )
outtree.Branch( 'ljet_pz',            b_ljet_pz )
outtree.Branch( 'ljet_pt',            b_ljet_pt )
outtree.Branch( 'ljet_eta',           b_ljet_eta )
outtree.Branch( 'ljet_phi',           b_ljet_phi )
outtree.Branch( 'ljet_e',             b_ljet_E  )
outtree.Branch( 'ljet_m',             b_ljet_m  )
outtree.Branch( 'ljet_tau2',          b_ljet_tau2 )
outtree.Branch( 'ljet_tau3',          b_ljet_tau4 )
outtree.Branch( 'ljet_tau32',         b_ljet_tau32 )
outtree.Branch( 'ljet_bmatch70_dR',   b_ljet_bmatch70_dR )
outtree.Branch( 'ljet_bmatch70',      b_ljet_bmatch70 )
##outtree.Branch( 'ljet_isTopTagged80', b_ljet_isTopTagged80 )
outtree.Branch( 'ljet_smoothedTopTaggerMassTau32_topTag80',  b_ljet_isTopTagged80 )
outtree.Branch( 'abcd16',             b_abcd16, 'abcd16/I' )

n_good = 0
ientry = 0
for ientry in range(n_entries):

  tree.GetEntry(ientry)

  if ( n_entries < 10 ) or ( (ientry+1) % int(float(n_entries)/10.)  == 0 ):
    perc = 100. * ientry / float(n_entries)
    print "INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc )

  b_mcChannelNumber[0] = 0 #tree.Event.ProcessID
  b_runNumber[0]       = 0
  b_eventNumber[0]     = ientry #int( Number )

  jets = []
  jets_n = 2 #len( tree.Event.Jet )
  for i in range(jets_n):

    jets += [ TLorentzVector() ]
    j = jets[-1]

    pT  = tree.GetLeaf("Jet.PT").GetValue(i)
    eta = tree.GetLeaf("Jet.Eta").GetValue(i)
    phi = tree.GetLeaf("Jet.Phi").GetValue(i)
    M   = tree.GetLeaf("Jet.Mass").GetValue(i)

    j.SetPtEtaPhiM( pT, eta, phi, M )
    j.index = i
    j.tau2 = 0.
    j.tau3 = 0.
    j.tau32 = 0.
    j.bmatch70_dR = 10.
    j.bmatch70 = -1
    j.isTopTagged80 = -1

  jets_n = len( jets )

#  if jets_n < 2: continue
#  if jets[0].Pt() < 500: continue
#  if jets[1].Pt() < 350: continue

  # Fill branches
  for i in range( min(5,jets_n) ):
     j = jets[i]

     b_ljet_px[i]    =  j.Px()*GeV
     b_ljet_py[i]    =  j.Py()*GeV
     b_ljet_pz[i]    =  j.Pz()*GeV
     b_ljet_pt[i]    =  j.Pt()*GeV
     b_ljet_eta[i]   =  j.Eta()
     b_ljet_phi[i]   =  j.Phi()
     b_ljet_E[i]     =  j.E()*GeV
     b_ljet_m[i]     =  j.M()*GeV
     b_ljet_tau2[i]  =  j.tau2
     b_ljet_tau4[i]  =  j.tau3
     b_ljet_tau32[i] =  j.tau32
     b_ljet_bmatch70_dR[i] = j.bmatch70_dR
     b_ljet_bmatch70[i]    = j.bmatch70
     b_ljet_isTopTagged80[i] =  j.isTopTagged80

  outtree.Fill()
  n_good += 1

outtree.Write()
outfile.Close()

f_good = 100. * n_good / n_entries
print "INFO: %i entries written (%.2f %%)" % ( n_good, f_good)

