#!/usr/bin/env python

import sys
import glob
import ctypes

from ROOT import *
import numpy as np
from keras.models import load_model
import cPickle as pickle
from models import mmd_loss, gauss_loss
#from models import chi2_loss, wasserstein_loss, gauss_loss

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro("AtlasUtils.C")
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)

rng = TRandom3()

GeV = 1.
TeV = 1e3
pi = 3.1415

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def Normalize(h, sf=1.0):
    if h == None:
        return

    A = h.Integral()
    if A == 0.:
        return

    h.Scale(sf / A)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def SetTH1FStyle(h, color=kBlack, linewidth=1, linestyle=kSolid, fillcolor=0, fillstyle=0, markerstyle=21, markersize=1.3, fill_alpha=0):
    '''Set the style with a long list of parameters'''

    h.SetLineColor(color)
    h.SetLineWidth(linewidth)
    h.SetLineStyle(linestyle)
    h.SetFillColor(fillcolor)
    h.SetFillStyle(fillstyle)
    h.SetMarkerStyle(markerstyle)
    h.SetMarkerColor(h.GetLineColor())
    h.SetMarkerSize(markersize)
    if fill_alpha > 0:
        h.SetFillColorAlpha(color, fill_alpha)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


model_filename = sys.argv[1]

# GAN/generator.mg5_dijet_ht500.ptcl.pt250.nominal.epoch_00000.h5
level = model_filename.split('/')[-1].split('.')[-5]
epoch = model_filename.split('/')[-1].split('.')[-2].split('_')[-1]
epoch = int(epoch)

n_examples = 50000
if len(sys.argv) > 2:
    n_examples = int(sys.argv[2])

dsid = "mg5_dijet_ht500"
preselection = "pt250"
systematic = "nominal"


#scaler_filename = "GAN_%s/scaler.%s.pkl" % (level,level)
scaler_filename = "lorentz/scaler.%s.pkl" % ( level )
print "INFO: loading scaler from", scaler_filename
with open(scaler_filename, "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

_h = {}
_h['ljet1_pt'] = TH1F(
    "ljet1_pt", ";Leading large-R jet p_{T} [GeV];Events / Bin Width", 20,  200.,  800)
_h['ljet1_eta'] = TH1F(
    "ljet1_eta", ";Leading large-R jet #eta;Events / Bin Width", 25, -2.5, 2.5)
_h['ljet1_E'] = TH1F(
    "ljet1_E",  ";Leading large-R jet E [GeV];Events / Bin Width", 25, 0., 1500)
_h['ljet1_m'] = TH1F(
    "ljet1_m",  ";Leading large-R jet m [GeV];Events / Bin Width", 20, 0., 200.)

_h['ljet2_pt'] = TH1F(
    "ljet2_pt", ";2nd leading large-R jet p_{T} [GeV];Events / Bin Width", 20,  200.,  600)
_h['ljet2_eta'] = TH1F(
    "ljet2_eta", ";2nd leading large-R jet #eta;Events / Bin Width", 25, -2.5, 2.5)
_h['ljet2_E'] = TH1F(
    "ljet2_E",  ";2nd leading large-R jet E [GeV];Events / Bin Width", 25, 0., 1500)
_h['ljet2_m'] = TH1F(
    "ljet2_m",  ";2nd leading large-R jet m [GeV];Events / Bin Width", 20, 0., 200.)

_h['jj_pt'] = TH1F(
    "jj_pt", ";Dijet system p_{T} [GeV];Events / Bin Width", 15, 0., 300)
_h['jj_eta'] = TH1F(
    "jj_eta", ";Dijet system #eta;Events / Bin Width", 30, -6.0, 6.0)
_h['jj_E'] = TH1F(
    "jj_E",  ";Dijet system E [GeV];Events / Bin Width", 15, 0., 3000)
_h['jj_m'] = TH1F(
    "jj_m",  ";Dijet system m [GeV];Events / Bin Width", 20, 0., 2.)
_h['jj_dM'] = TH1F(
    "jj_dM",   ";Dijet system #Delta M;Events / Bin Width",  20, -200, 200 )

_h['jj_dPhi'] = TH1F(
    "jj_dPhi", ";Dijet system #Delta#phi;Events / Bin Width", 16, pi/2., pi)
_h['jj_dEta'] = TH1F(
    "jj_dEta", ";Dijet system #Delta#eta;Events / Bin Width", 30, -3., 3.)
_h['jj_dR'] = TH1F(
    "jj_dR",   ";Dijet system #Delta R;Events / Bin Width",   15, 2., 5)

mc_filename = "histograms/histograms.%s.%s.%s.MC.root" % (
    dsid, level, preselection)
f_mc = TFile.Open(mc_filename)
_h_mc = {}
_h_mc['ljet1_pt'] = f_mc.Get('ljet1_pt')
_h_mc['ljet1_eta'] = f_mc.Get('ljet1_eta')
_h_mc['ljet1_E'] = f_mc.Get('ljet1_E')
_h_mc['ljet1_m'] = f_mc.Get('ljet1_m')
_h_mc['ljet2_pt'] = f_mc.Get('ljet2_pt')
_h_mc['ljet2_eta'] = f_mc.Get('ljet2_eta')
_h_mc['ljet2_E'] = f_mc.Get('ljet2_E')
_h_mc['ljet2_m'] = f_mc.Get('ljet2_m')
_h_mc['jj_pt'] = f_mc.Get('jj_pt')
_h_mc['jj_eta'] = f_mc.Get('jj_eta')
_h_mc['jj_E'] = f_mc.Get('jj_E')
_h_mc['jj_m'] = f_mc.Get('jj_m')
_h_mc['jj_dEta'] = f_mc.Get('jj_dEta')
_h_mc['jj_dPhi'] = f_mc.Get('jj_dPhi')
_h_mc['jj_dR'] = f_mc.Get('jj_dR')
_h_mc['jj_dM'] = f_mc.Get('jj_dM')

for h in _h_mc.values():
    SetTH1FStyle(h,  color=kGray+2, fillstyle=1001,
                 fillcolor=kGray, linewidth=3, markersize=0)
    Normalize(h)
    h.SetMinimum(0.)
    h.SetMaximum(1.3*h.GetMaximum())

for h in _h.values():
    SetTH1FStyle(h, color=kBlack, markersize=0,
                 markerstyle=20, linewidth=3)

c = TCanvas("c", "C", 1200, 1200)
c.Divide(4, 4)

generator = load_model(model_filename,
                       custom_objects={'mmd_loss': mmd_loss})

decoder_filename = "lorentz/model_decoder.%s.h5" % (level)
decoder = load_model( decoder_filename )

GAN_noise_size = generator.layers[0].input_shape[1]
n_latent       = decoder.layers[0].input_shape[1]
n_features     = decoder.layers[-1].output_shape[1]

print "INFO: decoder: (%i) -> (%i)" % ( n_latent, n_features )

X_noise = np.random.uniform(
    0, 1, size=[n_examples, GAN_noise_size])
events = generator.predict(X_noise)
events = decoder.predict(events)
events = scaler.inverse_transform(events)

_h['ljet1_pt'].Reset()
_h['ljet1_eta'].Reset()
_h['ljet1_E'].Reset()
_h['ljet1_m'].Reset()

_h['ljet2_pt'].Reset()
_h['ljet2_eta'].Reset()
_h['ljet2_E'].Reset()
_h['ljet2_m'].Reset()

_h['jj_pt'].Reset()
_h['jj_eta'].Reset()
_h['jj_E'].Reset()
_h['jj_m'].Reset()

for i in range(n_examples):

    #    "ljet1_pt", "ljet1_eta", "ljet1_M",
    #    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M",
    #    "jj_pt",    "jj_eta",    "jj_phi",    "jj_M",
    #    "jj_dPhi",  "jj_dEta",  "jj_dR",

    lj1 = TLorentzVector()
    lj1.SetPtEtaPhiM(events[i][0],
                     events[i][1],
                     0.,
                     events[i][2])

    lj2 = TLorentzVector()
    lj2.SetPtEtaPhiM(events[i][3],
                     events[i][4],
                     events[i][5],
                     events[i][6])

    # flip and rotate phi
    if rng.Uniform() > 0.5:
        lj2.SetPtEtaPhiM(lj2.Pt(), lj2.Eta(), -lj2.Phi(), lj2.M())

    phi = rng.Uniform(-TMath.Pi(), TMath.Pi())
    lj1.RotateZ(phi)
    lj2.RotateZ(phi)

    if lj1.Pt() < lj2.Pt():
        continue
    if lj1.Pt() < 250:
        continue
    if lj2.Pt() < 250:
        continue

    jj = lj1+lj2
    jj.dEta = lj1.Eta() - lj2.Eta()
    jj.dPhi = lj1.DeltaPhi( lj2 )
    jj.dR   = lj1.DeltaR( lj2 )
    jj.dM   = lj1.M() - lj2.M()

    _h['ljet1_pt'].Fill(lj1.Pt()/GeV)
    _h['ljet1_eta'].Fill(lj1.Eta())
    _h['ljet1_E'].Fill(lj1.E()/GeV)
    _h['ljet1_m'].Fill(lj1.M()/GeV)

    _h['ljet2_pt'].Fill(lj2.Pt()/GeV)
    _h['ljet2_eta'].Fill(lj2.Eta())
    _h['ljet2_E'].Fill(lj2.E()/GeV)
    _h['ljet2_m'].Fill(lj2.M()/GeV)

    _h['jj_pt'].Fill(jj.Pt()/GeV)
    _h['jj_eta'].Fill(jj.Eta())
    _h['jj_E'].Fill(jj.E()/GeV)
    _h['jj_m'].Fill(jj.M()/TeV)
    _h['jj_dM'].Fill(jj.dM/GeV)

    _h['jj_dEta'].Fill( jj.dEta )
    _h['jj_dPhi'].Fill( abs(jj.dPhi) )
    _h['jj_dR'].Fill( jj.dR )

for h in _h.values():
    Normalize(h)

def PrintChi2(hname):
    chi2 = Double(0.)
    ndf = ctypes.c_int(0)
    igood = ctypes.c_int(0)
    #chi2 = _h_mc[hname].Chi2Test(_h[hname], "WW CHI2/NDF")
    _h_mc[hname].Chi2TestX(_h[hname], chi2, ndf, igood, "WW")
    ndf = ndf.value
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextSize(0.04)
    txt = "#chi^{2}/NDF = %.1f/%i = %.1f" % (chi2, ndf, chi2/ndf)
    l.DrawLatex(0.3, 0.87, txt)

    return chi2, ndf


chi2_tot = 0.
ndf_tot = 0
c.cd(1)
# gPad.SetLogy(1)
#_h_mc['ljet1_pt'].SetMinimum(1e-2)
_h_mc['ljet1_pt'].Draw("h")
_h['ljet1_pt'].Draw("h same")
chi2, ndf = PrintChi2('ljet1_pt')
chi2_tot += chi2
ndf_tot += ndf

c.cd(2)
_h_mc['ljet1_eta'].Draw("h")
_h['ljet1_eta'].Draw("h same")
chi2, ndf = PrintChi2('ljet1_eta')
chi2_tot += chi2
ndf_tot += ndf

c.cd(3)
_h_mc['ljet1_E'].Draw("h")
_h['ljet1_E'].Draw("h same")
chi2, ndf = PrintChi2('ljet1_E')
chi2_tot += chi2
ndf_tot += ndf

c.cd(4)
_h_mc['ljet1_m'].Draw("h")
_h['ljet1_m'].Draw("h same")
chi2, ndf = PrintChi2('ljet1_m')
chi2_tot += chi2
ndf_tot += ndf

#################
# 2nd leading jet

c.cd(5)
#_h_mc['ljet2_pt'].SetMinimum(1e-2)
# gPad.SetLogy(1)
_h_mc['ljet2_pt'].Draw("h")
_h['ljet2_pt'].Draw("h same")
chi2, ndf = PrintChi2('ljet2_pt')
chi2_tot += chi2
ndf_tot += ndf

c.cd(6)
_h_mc['ljet2_eta'].Draw("h")
_h['ljet2_eta'].Draw("h same")
chi2, ndf = PrintChi2('ljet2_eta')
chi2_tot += chi2
ndf_tot += ndf

c.cd(7)
_h_mc['ljet2_E'].Draw("h")
_h['ljet2_E'].Draw("h same")
chi2, ndf = PrintChi2('ljet2_E')
chi2_tot += chi2
ndf_tot += ndf

c.cd(8)
_h_mc['ljet2_m'].Draw("h")
_h['ljet2_m'].Draw("h same")
chi2, ndf = PrintChi2('ljet2_m')
chi2_tot += chi2
ndf_tot += ndf

####################
# Dijet system

c.cd(9)
#_h_mc['jj_pt'].SetMinimum(1e-2)
# gPad.SetLogy(1)
_h_mc['jj_pt'].Draw("h")
_h['jj_pt'].Draw("h same")
chi2, ndf = PrintChi2('jj_pt')
chi2_tot += chi2
ndf_tot += ndf

c.cd(10)
_h_mc['jj_eta'].Draw("h")
_h['jj_eta'].Draw("h same")
chi2, ndf = PrintChi2('jj_eta')
chi2_tot += chi2
ndf_tot += ndf

c.cd(11)
_h_mc['jj_E'].Draw("h")
_h['jj_E'].Draw("h same")
chi2, ndf = PrintChi2('jj_E')
chi2_tot += chi2
ndf_tot += ndf

c.cd(12)
_h_mc['jj_m'].Draw("h")
_h['jj_m'].Draw("h same")
chi2, ndf = PrintChi2('jj_m')
chi2_tot += chi2
ndf_tot += ndf


#######################
# Angular variables

c.cd(13)
_h_mc['jj_dEta'].Draw("h")
_h['jj_dEta'].Draw("h same")
chi2, ndf = PrintChi2('jj_dEta')
chi2_tot += chi2
ndf_tot += ndf

c.cd(14)
_h_mc['jj_dPhi'].Draw("h")
_h['jj_dPhi'].Draw("h same")
chi2, ndf = PrintChi2('jj_dPhi')
chi2_tot += chi2
ndf_tot += ndf

c.cd(15)
_h_mc['jj_dR'].Draw("h")
_h['jj_dR'].Draw("h same")
chi2, ndf = PrintChi2('jj_dR')
chi2_tot += chi2
ndf_tot += ndf

c.cd(16)
_h_mc['jj_dM'].Draw("h")
_h['jj_dM'].Draw("h same")
chi2, ndf = PrintChi2('jj_dM')
chi2_tot += chi2
ndf_tot += ndf

c.cd()

imgname = "img/%s/training_%s_%s_%s_epoch_%05i.png" % (
    level, dsid, level, preselection, epoch)
c.SaveAs(imgname)

chi2_o_ndf = chi2_tot / ndf_tot
print "RESULT: epoch %i : chi2/ndf = %.1f / %i = %.1f" % (
    epoch, chi2_tot, ndf_tot, chi2_o_ndf)
