#!/usr/bin/env python

import os, sys
from ROOT import *
from array import array

GeV = 1e3
TeV = 1e6

gROOT.SetBatch(1)

filelistname = sys.argv[1]

n_events_max = -1
if len(sys.argv) > 2: n_events_max = int( sys.argv[2] )

tree_name = "nominal"
tree = TChain( tree_name, tree_name )
f = open( filelistname, 'r' )
for fname in f.readlines():
   fname = fname.strip()
   tree.AddFile( fname )
n_entries = tree.GetEntries()
print "INFO: entries found:", n_entries

outfilename = "histograms/histograms." + filelistname.split("/")[-1].replace( ".txt", ".root" ) 
outfile = TFile.Open( outfilename, "RECREATE" )

_h = {}
_h['ljet1_px'] = TH1F( "ljet1_px", ";Leading large-R jet p_{x} [GeV];Events / Bin Width", 40, -1000, 1000 )
_h['ljet1_py'] = TH1F( "ljet1_py", ";Leading large-R jet p_{y} [GeV];Events / Bin Width", 40, -1000, 1000 )
_h['ljet1_pz'] = TH1F( "ljet1_pz", ";Leading large-R jet p_{z} [GeV];Events / Bin Width", 40, -2000, 2000 )
_h['ljet1_pt'] = TH1F( "ljet1_pt", ";Leading large-R jet p_{T} [GeV];Events / Bin Width", 30,    0., 1500 )
_h['ljet1_eta'] = TH1F( "ljet1_eta", ";Leading large-R jet #eta;Events / Bin Width", 40, -2.0, 2.0 )
_h['ljet1_phi'] = TH1F( "ljet1_phi", ";Leading large-R jet #phi;Events / Bin Width", 32, 0., 3.1415 )
_h['ljet1_E']  = TH1F( "ljet1_E",  ";Leading large-R jet E [GeV];Events / Bin Width", 20, 0., 2000 )
_h['ljet1_m']  = TH1F( "ljet1_m",  ";Leading large-R jet m [GeV];Events / Bin Width", 30, 0., 300. )
_h['ljet1_tau2']  = TH1F( "ljet1_tau2",  ";Leading large-R jet #tau_{2};Events / Bin Width",  20, 0., 0.1 )
_h['ljet1_tau3']  = TH1F( "ljet1_tau3",  ";Leading large-R jet #tau_{3};Events / Bin Width",  20, 0., 0.1 )
_h['ljet1_tau32'] = TH1F( "ljet1_tau32", ";Leading large-R jet #tau_{32};Events / Bin Width", 20, 0., 1. )

_h['ljet2_px'] = TH1F( "ljet2_px", ";2nd leading large-R jet p_{x} [GeV];Events / Bin Width", 40, -1000, 1000 )
_h['ljet2_py'] = TH1F( "ljet2_py", ";2nd leading large-R jet p_{y} [GeV];Events / Bin Width", 40, -1000, 1000 )
_h['ljet2_pz'] = TH1F( "ljet2_pz", ";2nd leading large-R jet p_{z} [GeV];Events / Bin Width", 40, -2000, 2000 )
_h['ljet2_pt'] = TH1F( "ljet2_pt", ";2nd leading large-R jet p_{T} [GeV];Events / Bin Width", 30,    0., 1500 )
_h['ljet2_eta'] = TH1F( "ljet2_eta", ";2nd leading large-R jet #eta;Events / Bin Width", 40, -2.0, 2.0 )
_h['ljet2_phi'] = TH1F( "ljet2_phi", ";2nd leading large-R jet #phi;Events / Bin Width", 32, 0., 3.1415 )
_h['ljet2_E']  = TH1F( "ljet2_E",  ";2nd leading large-R jet E [GeV];Events / Bin Width", 20, 0., 2000 )
_h['ljet2_m']  = TH1F( "ljet2_m",  ";2nd leading large-R jet m [GeV];Events / Bin Width", 30, 0., 300. )
_h['ljet2_tau2']  = TH1F( "ljet2_tau2",  ";2nd leading large-R jet #tau_{2};Events / Bin Width",  20, 0., 0.1 )
_h['ljet2_tau3']  = TH1F( "ljet2_tau3",  ";2nd leading large-R jet #tau_{3};Events / Bin Width",  20, 0., 0.1 )
_h['ljet2_tau32'] = TH1F( "ljet2_tau32", ";2nd leading large-R jet #tau_{32};Events / Bin Width", 20, 0., 1. )

_h['jj_px'] = TH1F( "jj_px", ";Dijet system p_{x} [GeV];Events / Bin Width", 40, -1000, 1000 )
_h['jj_py'] = TH1F( "jj_py", ";Dijet system p_{y} [GeV];Events / Bin Width", 40, -1000, 1000 )
_h['jj_pz'] = TH1F( "jj_pz", ";Dijet system p_{z} [GeV];Events / Bin Width", 40, -2000, 2000 )
_h['jj_pt'] = TH1F( "jj_pt", ";Dijet system p_{T} [GeV];Events / Bin Width", 30,    0., 1500 )
_h['jj_eta'] = TH1F( "jj_eta", ";Dijet system #eta;Events / Bin Width", 40, -2.0, 2.0 )
_h['jj_phi'] = TH1F( "jj_phi", ";Dijet system #phi;Events / Bin Width", 32, 0., 3.1415 )
_h['jj_E']  = TH1F( "jj_E",  ";Dijet system E [GeV];Events / Bin Width", 20, 0., 2000 )
_h['jj_m']  = TH1F( "jj_m",  ";Dijet system m [GeV];Events / Bin Width", 30, 0., 300. )
_h['jj_dPhi'] = TH1F( "jj_dPhi", ";Dijet system #Delta#phi;Events / Bin Width", 20, 0., 3.1415 )
_h['jj_dEta'] = TH1F( "jj_dEta", ";Dijet system #Delta#eta;Events / Bin Width", 30, -3., 3. )
_h['jj_dR']   = TH1F( "jj_dR",   ";Dijet system #Delta R;Events / Bin Width",   25, 0., 5 )

_h['ljet1_E_vs_pt']  = TH2F( "ljet1_E_vs_pt",  ";Leading large-R jet p_{T} [GeV];Leading large-R jet E [GeV]", 20, 0., 2000, 50, 0., 2000 )
_h['ljet1_m_vs_pt']  = TH2F( "ljet1_m_vs_pt",  ";Leading large-R jet p_{T} [GeV];Leading large-R jet m [GeV]", 20, 0., 2000, 30, 0., 300. )
_h['ljet1_m_vs_eta'] = TH2F( "ljet1_m_vs_eta", ";Leading large-R jet #eta;Leading large-R jet m [GeV]", 40, -2.0, 2.0, 30, 0., 300. )

_h['ljet2_E_vs_pt']  = TH2F( "ljet2_E_vs_pt",  ";2nd leading large-R jet p_{T} [GeV];2nd leading large-R jet E [GeV]", 50, 0., 2000, 50, 0., 2000 )
_h['ljet2_m_vs_pt']  = TH2F( "ljet2_m_vs_pt",  ";2nd leading large-R jet p_{T} [GeV];2nd leading large-R jet m [GeV]", 50, 0., 2000, 30, 0., 300. )
_h['ljet2_m_vs_eta'] = TH2F( "ljet2_m_vs_eta", ";2nd leading large-R jet #eta;2nd leading large-R jet m [GeV]", 40, -2.0, 2.0, 30, 0., 300. )

_h['ljet2_pt_vs_ljet1_pt'] = TH2F( "ljet1_pt_vs_ljet2_pt", ";Leading large-R jet p_{T} [GeV];2nd leading large-R jet p_{T} [GeV]", 50, 0., 2000, 50, 0., 2000 )
_h['ljet2_m_vs_ljet1_m']   = TH2F( "ljet2_m_vs_ljet1_m",   ";Leading large-R jet m [GeV];2nd leading large-R jet m [GeV]", 30, 0., 300., 30, 0., 300. )

for h in _h.values(): h.Sumw2()

if n_events_max > 0: n_entries = min( n_events_max, n_entries )
print "INFO: starting event loop:", n_entries

for ientry in range(n_entries):

    if ( n_entries < 10 ) or ( (ientry+1) % int(float(n_entries)/10.)  == 0 ):
        perc = 100. * ientry / float(n_entries)
        print "INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc )

    tree.GetEntry( ientry )

    w = 1.0
    #w = tree.

    ljets = []
    for k in range(len(tree.ljet_pt)):
        if tree.ljet_pt[k] < 250*GeV: continue
        if abs(tree.ljet_eta[k]) > 2.0: continue
        ljets += [ TLorentzVector() ]
        lj = ljets[-1]
        lj.SetPtEtaPhiM( tree.ljet_pt[k], tree.ljet_eta[k], tree.ljet_phi[k], tree.ljet_m[k] )            
        lj.tau2  = tree.ljet_tau2[k]
        lj.tau3  = tree.ljet_tau3[k]
        lj.tau32 = tree.ljet_tau32[k]
        try:
            lj.topTag = tree.isTopTagged80[k]
        except:
            lj.topTag = tree.ljet_smoothedTopTaggerMassTau32_topTag80[k]
        #lj.sd12  = tree.ljet_sd12[k]
    ljets_n = len( ljets )
   
    if ljets_n < 2: continue
    #ljets.sort( key=lambda jet: jet.Pt(), reverse=True )
    
    lj1 = ljets[0]
    lj2 = ljets[1]

    if lj1.Pt() < 500*GeV: continue
    if lj2.Pt() < 350*GeV: continue

    jj = lj1 + lj2
    jj.dPhi = lj1.DeltaPhi( lj2 )
    jj.dEta = lj1.Eta() - lj2.Eta()
    jj.dR   = lj1.DeltaR( lj2 )

    #abcd16 = tree.abcd16

    _h['ljet1_px'].Fill( lj1.Px()/GeV, w )
    _h['ljet1_py'].Fill( lj1.Py()/GeV, w )
    _h['ljet1_pz'].Fill( lj1.Pz()/GeV, w )
    _h['ljet1_pt'].Fill( lj1.Pt()/GeV, w )
    _h['ljet1_eta'].Fill( lj1.Eta(), w )
    _h['ljet1_phi'].Fill( lj1.Phi(), w )
    _h['ljet1_E'].Fill( lj1.E()/GeV, w )
    _h['ljet1_m'].Fill( lj1.M()/GeV, w )
    _h['ljet1_tau2'].Fill( lj1.tau2, w )
    _h['ljet1_tau3'].Fill( lj1.tau3, w )
    _h['ljet1_tau32'].Fill( lj1.tau32, w )

    _h['ljet2_px'].Fill( lj2.Px()/GeV, w )
    _h['ljet2_py'].Fill( lj2.Py()/GeV, w )
    _h['ljet2_pz'].Fill( lj2.Pz()/GeV, w )
    _h['ljet2_pt'].Fill( lj2.Pt()/GeV, w )
    _h['ljet2_eta'].Fill( lj2.Eta(), w )
    _h['ljet2_phi'].Fill( lj2.Phi(), w )
    _h['ljet2_E'].Fill( lj2.E()/GeV, w )
    _h['ljet2_m'].Fill( lj2.M()/GeV, w )
    _h['ljet2_tau2'].Fill( lj2.tau2, w )
    _h['ljet2_tau3'].Fill( lj2.tau3, w )
    _h['ljet2_tau32'].Fill( lj2.tau32, w )

    _h['jj_px'].Fill( jj.Px()/GeV, w )
    _h['jj_py'].Fill( jj.Py()/GeV, w )
    _h['jj_pz'].Fill( jj.Pz()/GeV, w )
    _h['jj_pt'].Fill( jj.Pt()/GeV, w )
    _h['jj_eta'].Fill( jj.Eta(), w )
    _h['jj_phi'].Fill( jj.Phi(), w )
    _h['jj_E'].Fill( jj.E()/GeV, w )
    _h['jj_m'].Fill( jj.M()/GeV, w )
    _h['jj_dPhi'].Fill( jj.dPhi, w )
    _h['jj_dEta'].Fill( jj.dEta, w )
    _h['jj_dR'].Fill( jj.dR, w )
    
    _h['ljet1_E_vs_pt'].Fill(  lj1.Pt()/GeV, lj1.E()/GeV, w )
    _h['ljet1_m_vs_pt'].Fill(  lj1.Pt()/GeV, lj1.M()/GeV, w )
    _h['ljet1_m_vs_eta'].Fill( lj1.Eta(),    lj1.M()/GeV, w )

    _h['ljet2_E_vs_pt'].Fill(  lj2.Pt()/GeV, lj2.E()/GeV, w )
    _h['ljet2_m_vs_pt'].Fill(  lj2.Pt()/GeV, lj2.M()/GeV, w )
    _h['ljet2_m_vs_eta'].Fill( lj2.Eta(),    lj2.M()/GeV, w )

    _h['ljet2_pt_vs_ljet1_pt'].Fill( lj1.Pt(), lj2.Pt(), w )
    _h['ljet2_m_vs_ljet1_m'].Fill( lj1.M(), lj2.M(), w )


outfile.Write()
outfile.Close()
print "INFO: Create output file:", outfile.GetName()
