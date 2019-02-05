#!/usr/bin/env python

import sys
import glob

from ROOT import *
import numpy as np
from keras.models import load_model
import cPickle as pickle

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro("AtlasUtils.C")
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)

rng = TRandom3()

GeV = 1.
TeV = 1e3

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


level = "ptcl"
if len(sys.argv) > 1:
    level = sys.argv[1]

n_examples = 50000
if len(sys.argv) > 2:
    n_examples = int(sys.argv[2])

dsid = "mg5_dijet_ht500"
preselection = "pt250"
systematic = "nominal"


scaler_filename = "GAN/scaler.%s.pkl" % level
print "INFO: loading scaler from", scaler_filename
with open(scaler_filename, "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

_h = {}
_h['ljet1_pt'] = TH1F(
    "ljet1_pt", ";Leading large-R jet p_{T} [GeV];Events / Bin Width", 40,  200.,  800)
_h['ljet1_eta'] = TH1F(
    "ljet1_eta", ";Leading large-R jet #eta;Events / Bin Width", 40, -2.0, 2.0)
_h['ljet1_m'] = TH1F(
    "ljet1_m",  ";Leading large-R jet m [GeV];Events / Bin Width", 40, 0., 200.)

_h['ljet2_pt'] = TH1F(
    "ljet2_pt", ";2nd leading large-R jet p_{T} [GeV];Events / Bin Width", 40,  200.,  800)
_h['ljet2_eta'] = TH1F(
    "ljet2_eta", ";2nd leading large-R jet #eta;Events / Bin Width", 40, -2.0, 2.0)
_h['ljet2_m'] = TH1F(
    "ljet2_m",  ";2nd leading large-R jet m [GeV];Events / Bin Width", 40, 0., 200.)

_h['jj_pt'] = TH1F(
    "jj_pt", ";Dijet system p_{T} [GeV];Events / Bin Width", 30,    0., 300)
_h['jj_eta'] = TH1F(
    "jj_eta", ";Dijet system #eta;Events / Bin Width", 60, -6.0, 6.0)
_h['jj_m'] = TH1F(
    "jj_m",  ";Dijet system m [GeV];Events / Bin Width", 40, 0., 2.)


mc_filename = "histograms/histograms.%s.%s.%s.MC.root" % (
    dsid, level, preselection)
f_mc = TFile.Open(mc_filename)
_h_mc = {}
_h_mc['ljet1_pt'] = f_mc.Get('ljet1_pt')
_h_mc['ljet1_eta'] = f_mc.Get('ljet1_eta')
_h_mc['ljet1_m'] = f_mc.Get('ljet1_m')
_h_mc['ljet2_pt'] = f_mc.Get('ljet2_pt')
_h_mc['ljet2_eta'] = f_mc.Get('ljet2_eta')
_h_mc['ljet2_m'] = f_mc.Get('ljet2_m')
_h_mc['jj_pt'] = f_mc.Get('jj_pt')
_h_mc['jj_eta'] = f_mc.Get('jj_eta')
_h_mc['jj_m'] = f_mc.Get('jj_m')

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
c.Divide(3, 3)

gen_filepaths = "GAN/generator.%s.%s.%s.%s.epoch_*.h5" % (
    dsid, level, preselection, systematic)
gen_filepaths = glob.glob(gen_filepaths)
print gen_filepaths

for model_filename in gen_filepaths:

    epoch = model_filename.split("/")[-1].split(".")[-2].split("_")[-1]
    epoch = int(epoch)
    print "INFO: epoch", epoch

   # model_filename = "GAN/generator.%s.%s.%s.%s.epoch_%05i.h5" % (
   #     dsid, level, preselection, systematic, epoch)

    generator = load_model(model_filename)

    GAN_noise_size = generator.layers[0].input_shape[1]

    X_noise = np.random.uniform(
        0, 1, size=[n_examples, GAN_noise_size])
    events = generator.predict(X_noise)
    events = scaler.inverse_transform(events)

    _h['ljet1_pt'].Reset()
    _h['ljet1_eta'].Reset()
    _h['ljet1_m'].Reset()

    _h['ljet2_pt'].Reset()
    _h['ljet2_eta'].Reset()
    _h['ljet2_m'].Reset()

    _h['jj_pt'].Reset()
    _h['jj_eta'].Reset()
    _h['jj_m'].Reset()

    for i in range(n_examples):

        lj1 = TLorentzVector()
        lj1.SetPtEtaPhiM(events[i][0], events[i][1], 0., events[i][2])

        lj2 = TLorentzVector()
        lj2.SetPtEtaPhiM(events[i][3], events[i][4],
                         events[i][5], events[i][6])

        phi = rng.Uniform(-TMath.Pi(), TMath.Pi())
        lj1.RotateZ(phi)
        lj2.RotateZ(phi)

        jj = lj1+lj2

        _h['ljet1_pt'].Fill(lj1.Pt()/GeV)
        _h['ljet1_eta'].Fill(lj1.Eta())
        _h['ljet1_m'].Fill(lj1.M()/GeV)

        _h['ljet2_pt'].Fill(lj2.Pt()/GeV)
        _h['ljet2_eta'].Fill(lj2.Eta())
        _h['ljet2_m'].Fill(lj2.M()/GeV)

        _h['jj_pt'].Fill(jj.Pt()/GeV)
        _h['jj_eta'].Fill(jj.Eta())
        _h['jj_m'].Fill(jj.M()/TeV)

    for h in _h.values():
        Normalize(h)

    def PrintChi2(hname):
        chi2 = _h_mc[hname].Chi2Test(_h[hname], "WW CHI2/NDF")
        l = TLatex()
        l.SetNDC()
        l.SetTextFont(42)
        l.SetTextSize(0.04)
        txt = "#chi^{2}/NDF = %.2f" % chi2
        l.DrawLatex(0.3, 0.85, txt)

    c.cd(1)
    _h_mc['ljet1_pt'].Draw("h")
    _h['ljet1_pt'].Draw("h same")
    PrintChi2('ljet1_pt')

    c.cd(2)
    _h_mc['ljet1_eta'].Draw("h")
    _h['ljet1_eta'].Draw("h same")
    PrintChi2('ljet1_eta')

    c.cd(3)
    _h_mc['ljet1_m'].Draw("h")
    _h['ljet1_m'].Draw("h same")
    PrintChi2('ljet1_m')

    c.cd(4)
    _h_mc['ljet2_pt'].Draw("h")
    _h['ljet2_pt'].Draw("h same")
    PrintChi2('ljet2_pt')

    c.cd(5)
    _h_mc['ljet2_eta'].Draw("h")
    _h['ljet2_eta'].Draw("h same")
    PrintChi2('ljet2_eta')

    c.cd(6)
    _h_mc['ljet2_m'].Draw("h")
    _h['ljet2_m'].Draw("h same")
    PrintChi2('ljet2_m')

    c.cd(7)
    _h_mc['jj_pt'].Draw("h")
    _h['jj_pt'].Draw("h same")
    PrintChi2('jj_pt')

    c.cd(8)
    _h_mc['jj_eta'].Draw("h")
    _h['jj_eta'].Draw("h same")
    PrintChi2('jj_eta')

    c.cd(9)
    _h_mc['jj_m'].Draw("h")
    _h['jj_m'].Draw("h same")
    PrintChi2('jj_m')

    c.cd()

    imgname = "img/training_%s_%s_%s_epoch_%05i.png" % (
        dsid, level, preselection, epoch)
    c.SaveAs(imgname)
