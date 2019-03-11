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

np.set_printoptions(precision=2, suppress=True, linewidth=300)


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


obs = "jj_m"

model_filename = sys.argv[1]

# GAN/generator.mg5_dijet_ht500.ptcl.pt250.nominal.epoch_00000.h5
level = model_filename.split('/')[-1].split('.')[-5]
epoch = model_filename.split('/')[-1].split('.')[-2].split('_')[-1]
epoch = int(epoch)

n_examples = 1000000
if len(sys.argv) > 2:
    n_examples = int(sys.argv[2])

dsid = "mg5_dijet_ht500"
preselection = "pt250"
systematic = "nominal"


scaler_filename = "GAN_%s/scaler.%s.pkl" % (level, level)
# scaler_filename = "lorentz/scaler.%s.pkl" % ( level ) # DCGAN_AE
print "INFO: loading scaler from", scaler_filename
with open(scaler_filename, "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

mc_filename = "histograms/histograms.%s.%s.%s.MC.root" % (
    dsid, level, preselection)
f_mc = TFile.Open(mc_filename)
h_mc = f_mc.Get(obs+"_tail")

h_gan = h_mc.Clone(obs+"_gan")
h_gan.Reset()

SetTH1FStyle(h_gan, color=kBlack, markersize=0,
             markerstyle=20, linewidth=3)

SetTH1FStyle(h_mc,  color=kGray+2, fillstyle=1001,
             fillcolor=kGray, linewidth=3, markersize=0)

generator = load_model(model_filename)

#decoder_filename = "lorentz/model_decoder.%s.h5" % (level)
#decoder = load_model(decoder_filename)

GAN_noise_size = generator.layers[0].input_shape[1]
n_features = generator.layers[-1].output_shape[1]
#n_latent = decoder.layers[0].input_shape[1]
#print "INFO: decoder: (%i) -> (%i)" % (n_latent, n_features)

X_noise = np.random.uniform(
    0, 1, size=[n_examples, GAN_noise_size])
events = generator.predict(X_noise)
#events = decoder.predict(events)
events = scaler.inverse_transform(events)

for i in range(n_examples):

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

    h_gan.Fill(jj.M()/TeV)

# Normalize(h_mc)
area_mc = h_mc.Integral()
Normalize(h_gan, area_mc)


def PrintChi2(h_mc, h_gan):
    chi2 = Double(0.)
    ndf = ctypes.c_int(0)
    igood = ctypes.c_int(0)
    #chi2 = _h_mc[hname].Chi2Test(_h[hname], "WW CHI2/NDF")
    h_mc.Chi2TestX(h_gan, chi2, ndf, igood, "WW")
    ndf = ndf.value

    return chi2, ndf


c = TCanvas("c", "C", 1600, 1200)

h_mc.Draw("h")

h_mc.SetMaximum(1e6)
gPad.SetLogy(True)

# fit_func = TF1(
#    "fit_func", "[0] * ( pow( 1-(x/13.), [1] ) ) * ( pow((x/13.),([2]+[3]*log((x/13.))+[4]*pow(log(x/13.),2) ) ) )")

fit_func = TF1(
    "fit_func", "[0] * ( pow( 1-(x/13.), [1] ) ) * ( pow((x/13.),[2]))")

fit_func.SetParameter(0, 100.)
fit_func.SetParLimits(0, 1e1, 1e5)

fit_func.SetParameter(1, 1e2)
#fit_func.SetParLimits(1, 1., 50.)

fit_func.SetParameter(2, 1e1)
#fit_func.SetParLimits(2, 0.1, 5.)

#fit_func.SetParameter(3, 1.)
#fit_func.SetParLimits(3, 0.1, 5.)

#fit_func.SetParameter(4, 0.1)
#fit_func.SetParLimits(4, 0.01, 1.)

fit_func.SetLineColor(kRed)

fit_res = h_mc.Fit("fit_func", "R S  ", "", 1., 10.)
chi2_fit = fit_res.Chi2()
ndf_fit = fit_res.Ndf()
chi2_o_ndf_fit = chi2_fit / ndf_fit
print "RESULT: 3-params fit: chi2=%.2f ndf=%i chi2/ndf=%.3f" % (
    chi2_fit, ndf_fit, chi2_o_ndf_fit)


h_gan.Draw("h same")

chi2_gan, ndf_gan = PrintChi2(h_mc, h_gan)

chi2_o_ndf_gan = chi2_gan / ndf_gan
print "RESULT: GAN: epoch %i : chi2/ndf = %.1f / %i = %.1f" % (
    epoch, chi2_gan, ndf_gan, chi2_o_ndf_gan)
l = TLatex()
l.SetNDC()
l.SetTextFont(42)
l.SetTextSize(0.03)
txt = "3p func: #chi^{2}/NDF = %.1f/%i = %.1f" % (
    chi2_fit, ndf_fit, chi2_o_ndf_fit)
l.DrawLatex(0.3, 0.87, txt)
txt = "GAN: #chi^{2}/NDF = %.1f/%i = %.1f" % (
    chi2_gan, ndf_gan, chi2_o_ndf_gan)
l.DrawLatex(0.3, 0.82, txt)

imgname = "img/%s/extrapolation_%s_%s_%s_%s_epoch_%05i.png" % (
    level, obs, dsid, level, preselection, epoch)
c.SaveAs(imgname)

# imgname = "img/%s/training_%s_%s_%s.gif++5" % (
#    level, dsid, level, preselection)
# c.Print(imgname)
