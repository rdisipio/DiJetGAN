#!/usr/bin/env python

import sys
import glob
import ctypes
import time

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


def RemoveBeforeCut(h, xmin):
    h2 = h.Clone(h.GetName()+"_chi2")
    nmin = h.FindBin(xmin)
    for i in range(h.GetNbinsX()):
        if i < nmin:
            h2.SetBinContent(i+1, 0.)
            h2.SetBinError(i+1, 0.)
    return h2

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
level = model_filename.split('/')[-1].split('.')[3]
epoch = model_filename.split('/')[-1].split('.')[-2].split('_')[-1]
epoch = int(epoch)
#epoch = 100000

n_examples = 2000
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

mc_filename_large = "histograms/histograms.%s.%s.%s.MC_large.root" % (
    dsid, level, preselection)
mc_filename_small = "histograms/histograms.%s.%s.%s.MC_small.root" % (
    dsid, level, preselection)

f_mc_large = TFile.Open(mc_filename_large)
h_mc_large = f_mc_large.Get(obs+"_tail").Clone(obs+"_mc_large")

f_mc_small = TFile.Open(mc_filename_small)
h_mc_small = f_mc_small.Get(obs+"_tail").Clone(obs+"_mc_small")

h_gan = h_mc_large.Clone(obs+"_gan")
h_gan.Reset()

SetTH1FStyle(h_gan, color=kBlack, markersize=0,
             markerstyle=20, linewidth=3)

SetTH1FStyle(h_mc_large,  color=kGray+2, fillstyle=1001,
             fillcolor=kGray, linewidth=3, markersize=0)
SetTH1FStyle(h_mc_small,  color=kGray+2, fillstyle=1001,
             fillcolor=kGray+1, linewidth=3, markersize=0, fill_alpha=0.5)

generator = load_model(model_filename)

#decoder_filename = "lorentz/model_decoder.%s.h5" % (level)
#decoder = load_model(decoder_filename)

GAN_noise_size = generator.layers[0].input_shape[1]
n_features = generator.layers[-1].output_shape[1]
#n_latent = decoder.layers[0].input_shape[1]
#print "INFO: decoder: (%i) -> (%i)" % (n_latent, n_features)

print "INFO: generating %i events..." % n_examples
X_noise = np.random.uniform(
    0, 1, size=[n_examples, GAN_noise_size])
time_start = time.time()
events = generator.predict(X_noise)
#events = decoder.predict(events)
events = scaler.inverse_transform(events)
time_end = time.time()
time_diff = time_end - time_start
print "INFO: generated %i events in %i sec" % (n_examples, time_diff)

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
m_jj_cut = 3.0
bin_xmin = h_mc_large.FindBin(m_jj_cut)
bin_xmax = h_mc_large.FindBin(10)
area_mc_large = h_mc_large.Integral(bin_xmin, bin_xmax)
area_gan = h_gan.Integral(bin_xmin, bin_xmax)
sf = area_mc_large / area_gan
#Normalize(h_gan, area_mc)
h_gan.Scale(sf)
Normalize(h_mc_small, area_mc_large)


c = TCanvas("c", "C", 1600, 1200)

h_mc_large.SetMinimum(1e-2)
h_mc_large.Draw("h")

h_mc_large.SetMaximum(1e6)
gPad.SetLogy(True)

# fit_func = TF1(
#    "fit_func", "[0] * ( pow( 1-(x/13.), [1] ) ) * ( pow((x/13.),([2]+[3]*log((x/13.))+[4]*pow(log(x/13.),2) ) ) )")

fit_func = TF1(
    "fit_func", "[0] * ( pow( 1-(x/13.), [1] ) ) / ( pow((x/13.),[2]+[3]*log(x/13.) ))")

fit_func.SetParameter(0, 100.)
fit_func.SetParLimits(0, 1e1, 1e4)

fit_func.SetParameter(1, 1e2)
#fit_func.SetParLimits(1, 1., 50.)

fit_func.SetParameter(2, 1e1)
#fit_func.SetParLimits(2, 0.1, 5.)

fit_func.SetParameter(3, 1.)
#fit_func.SetParLimits(3, 0.1, 5.)

#fit_func.SetParameter(4, 0.1)
#fit_func.SetParLimits(4, 0.01, 1.)

fit_func.SetLineColor(kRed)

# Fit function on small MC sample
fit_res = h_mc_large.Fit("fit_func", "R S", "", m_jj_cut, 10.)
h_fit = h_mc_large.GetFunction("fit_func")
chi2_fit = fit_res.Chi2()
ndf_fit = fit_res.Ndf()
chi2_o_ndf_fit = chi2_fit / ndf_fit
print "RESULT: 4-params fit: chi2=%.2f ndf=%i chi2/ndf=%.3f" % (
    chi2_fit, ndf_fit, chi2_o_ndf_fit)

g_unc = TGraphErrors()


#h_mc_small.Draw("h same")
h_gan.Draw("h same")
# h_fit.Draw("same")

cut_line = TLine()
cut_line.SetLineWidth(3)
cut_line.SetLineStyle(kDashed)
cut_line.SetLineColor(kGray+3)
cut_line.DrawLine(m_jj_cut, 0., m_jj_cut, 1e5)

h_gan_chi2 = RemoveBeforeCut(h_gan, m_jj_cut)
h_mc_chi2 = RemoveBeforeCut(h_mc_large, m_jj_cut)

chi2_o_ndf_gan = h_mc_chi2.Chi2Test(h_gan_chi2, "UU NORM CHI2/NDF")
chi2_gan = h_mc_chi2.Chi2Test(h_gan_chi2, "UU NORM CHI2")
ndf_gan = int(chi2_gan / chi2_o_ndf_gan)

fit_func.FixParameter(0, h_fit.GetParameter(0))
fit_func.FixParameter(1, h_fit.GetParameter(1))
fit_func.FixParameter(2, h_fit.GetParameter(2))
fit_func.FixParameter(3, h_fit.GetParameter(3))

fit_res_fx = h_gan.Fit("fit_func", "R S B", "", m_jj_cut, 10.)
h_fit_fx = h_gan.GetFunction("fit_func")
chi2_fx = fit_res_fx.Chi2()
ndf_fx = fit_res_fx.Ndf()
chi2_o_ndf_fx = chi2_fx / ndf_fx
print "RESULT: fx: epoch %i : chi2/ndf = %.1f / %i = %.2f" % (
    epoch, chi2_fx, ndf_fx, chi2_o_ndf_fx)

chi2_o_ndf_gan = chi2_gan / ndf_gan
print "RESULT: GAN: epoch %i : chi2/ndf = %.1f / %i = %.1f" % (
    epoch, chi2_gan, ndf_gan, chi2_o_ndf_gan)
l = TLatex()
l.SetNDC()
l.SetTextFont(42)
l.SetTextSize(0.03)
txt_gan = "#chi^{2}/dof = %.1f/%i = %.2f" % (
    chi2_gan, ndf_gan, chi2_o_ndf_gan)
#l.DrawLatex(0.3, 0.87, txt_gan)
txt_fit = "#chi^{2}/dof = %.1f/%i = %.2f" % (
    chi2_fit, ndf_fit, chi2_o_ndf_fit)
#l.DrawLatex(0.3, 0.83, txt_fit)

x2 = h_gan.Chisquare(fit_func, "R")
dof = h_gan.FindBin(10.) - h_gan.FindBin(m_jj_cut)
print "chisquare/dof = %f / %i = %.2f" % (x2, dof, x2/dof)

h_dummy = TH1F()
leg = TLegend(0.50, 0.87, 0.70, 0.87)
leg.SetNColumns(1)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetTextSize(0.04)
leg.AddEntry(h_mc_large, "MG5 + Py8", "f")
leg.AddEntry(h_fit, "4p fit: "+txt_fit, "l")
#leg.AddEntry(h_dummy, txt_fit, "")
leg.AddEntry(h_gan, "GAN: "+txt_gan, "f")
#leg.AddEntry(h_dummy, txt_gan, "")
leg.SetY1(leg.GetY1() - 0.05 * leg.GetNRows())
leg.Draw()


gPad.RedrawAxis()


def DivideBy(h1, h2):
    n = h1.GetNbinsX()
    for i in range(n):
        y1 = h1.GetBinContent(i+1)
        dy1 = h1.GetBinError(i+1)
        y2 = h2.GetBinContent(i+1)
        if y2 == 0.:
            continue
        h1.SetBinContent(i+1, y1/y2)
        h1.SetBinError(i+1, dy1/y2)


#h_r_gan = h_gan.Clone("r_gan")
#DivideBy(h_r_gan, h_mc_large)
# h_r_gan.SetMinimum(0.)
# h_r_gan.SetMaximum(2.)
#h_r_gan.GetXaxis().SetRangeUser(m_jj_cut, 10)
# h_r_gan.Print("all")

#subpad = TPad("subpad", "", 0.6, 0.4, 0.95, 0.65)
# subpad.Draw()
# subpad.cd()
# h_r_gan.Draw("h")

imgname = "img/%s/extrapolation_%s_%s_%s_%s_epoch_%05i.png" % (
    level, obs, dsid, level, preselection, epoch)
c.SaveAs(imgname)

# imgname = "img/%s/training_%s_%s_%s.gif++5" % (
#    level, dsid, level, preselection)
# c.Print(imgname)
