#!/usr/bin/env python

import os
import sys

from ROOT import *
from math import sqrt, pow, log

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro("AtlasUtils.C")
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def SetHistogramStyle(h, color=kBlack, linewidth=1, fillcolor=0, fillstyle=0, markerstyle=21, markersize=1.3, fill_alpha=0):
    '''Set the style with a long list of parameters'''

    h.SetLineColor(color)
    h.SetLineWidth(linewidth)
    h.SetFillColor(fillcolor)
    h.SetFillStyle(fillstyle)
    h.SetMarkerStyle(markerstyle)
    h.SetMarkerColor(h.GetLineColor())
    h.SetMarkerSize(markersize)
    if fill_alpha > 0:
        h.SetFillColorAlpha(color, fill_alpha)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


dsid = "mg5_dijet_ht500"
preselection = "incl"

if len(sys.argv) > 1:
    dsid = sys.argv[1]

infilename = "GAN/training_history.%s.rnn.GAN.%s.nominal.root" % (
    dsid, preselection)
infile = TFile.Open(infilename)

d_loss_mean = infile.Get("d_loss_mean")
g_loss_mean = infile.Get("g_loss_mean")

d_loss_orig = infile.Get("d_loss_orig")
d_loss_r_orig = infile.Get("d_loss_r_orig")
d_loss_f_orig = infile.Get("d_loss_f_orig")
g_loss_orig = infile.Get("g_loss_orig")
d_acc_orig = infile.Get("d_acc_orig")
d_acc_f_orig = infile.Get("d_acc_f_orig")
d_acc_r_orig = infile.Get("d_acc_r_orig")

n_epochs = d_loss_orig.GetN()
frame_d_loss = TH1F(
    "frame_d_loss", ";Training Epoch;Discriminator Loss", 10, 0, n_epochs)
frame_g_loss = TH1F(
    "frame_g_loss", ";Training Epoch;Generator Loss", 10, 0, n_epochs)
frame_d_acc = TH1F(
    "frame_d_acc",  ";Training Epoch;Discriminator Accuracy", 10, 0, n_epochs)

hmax = 1.6
frame_d_loss.SetMaximum(hmax)
frame_d_loss.SetMinimum(0.)
frame_d_loss.GetXaxis().SetTitleSize(0.07)
frame_d_loss.GetYaxis().SetTitleSize(0.07)
frame_d_loss.GetXaxis().SetLabelSize(0.07)
frame_d_loss.GetYaxis().SetLabelSize(0.07)
frame_d_loss.GetYaxis().SetTitleOffset(0.5)

hmax = 2.0
frame_d_acc.SetMaximum(hmax)
frame_d_acc.SetMinimum(0.)
frame_d_acc.GetXaxis().SetTitleSize(0.07)
frame_d_acc.GetYaxis().SetTitleSize(0.07)
frame_d_acc.GetXaxis().SetLabelSize(0.07)
frame_d_acc.GetYaxis().SetLabelSize(0.07)
frame_d_acc.GetYaxis().SetTitleOffset(0.5)

hmax = 1.3*TMath.MaxElement(g_loss_mean.GetN(), g_loss_mean.GetY())
frame_g_loss.SetMaximum(hmax)
frame_g_loss.SetMinimum(0.)
frame_g_loss.GetXaxis().SetTitleSize(0.07)
frame_g_loss.GetYaxis().SetTitleSize(0.07)
frame_g_loss.GetXaxis().SetLabelSize(0.07)
frame_g_loss.GetYaxis().SetLabelSize(0.07)
frame_g_loss.GetYaxis().SetTitleOffset(0.5)

SetHistogramStyle(d_loss_f_orig, color=kRed, linewidth=3)
SetHistogramStyle(d_loss_r_orig, color=kBlue, linewidth=3)
SetHistogramStyle(d_loss_orig,   color=kBlack, linewidth=3)

SetHistogramStyle(d_acc_f_orig, color=kRed, linewidth=3)
SetHistogramStyle(d_acc_r_orig, color=kBlue, linewidth=3)

SetHistogramStyle(g_loss_orig, color=kRed, linewidth=3)

SetHistogramStyle(d_loss_mean, color=kBlack, linewidth=3)
SetHistogramStyle(g_loss_mean, color=kBlack, linewidth=3)

c = TCanvas("C", "C", 1800, 1600)

gPad.Divide(1, 3)
gPad.SetLeftMargin(0.05)
gPad.SetTopMargin(0.05)
gPad.SetBottomMargin(0.10)
gPad.SetRightMargin(0.05)

# Loss (orig)
c.cd(1)
frame_d_loss.Draw()
d_loss_f_orig.Draw("l same")
d_loss_r_orig.Draw("l same")
d_loss_orig.Draw("l same")

leg1 = TLegend(0.30, 0.90, 0.80, 0.90)
leg1.SetNColumns(3)
leg1.SetFillColor(0)
leg1.SetFillStyle(0)
leg1.SetBorderSize(0)
leg1.SetTextFont(42)
leg1.SetTextSize(0.10)
leg1.AddEntry(d_loss_orig,   "Average loss", "l")
leg1.AddEntry(d_loss_r_orig, "Real loss", "l")
leg1.AddEntry(d_loss_f_orig, "Fake loss", "l")
leg1.SetY1(leg1.GetY1() - 0.10 * leg1.GetNRows())
leg1.Draw()

gPad.RedrawAxis()

# Acc (orig)
c.cd(2)
frame_d_acc.Draw()
d_acc_f_orig.Draw("l same")
d_acc_r_orig.Draw("l same")
d_acc_orig.Draw("l same")

leg2 = TLegend(0.30, 0.90, 0.80, 0.90)
leg2.SetNColumns(3)
leg2.SetFillColor(0)
leg2.SetFillStyle(0)
leg2.SetBorderSize(0)
leg2.SetTextFont(42)
leg2.SetTextSize(0.10)
leg2.AddEntry(d_acc_orig,   "Average acc", "l")
leg2.AddEntry(d_acc_r_orig, "Real acc", "l")
leg2.AddEntry(d_acc_f_orig, "Fake acc", "l")
leg2.SetY1(leg1.GetY1() - 0.05 * leg1.GetNRows())
leg2.Draw()

gPad.RedrawAxis()

# Generator loss
c.cd(3)
frame_g_loss.Draw()

g_loss_mean.Draw("l same")

leg6 = TLegend(0.65, 0.90, 0.80, 0.90)
leg6.SetFillColor(0)
leg6.SetFillStyle(0)
leg6.SetBorderSize(0)
leg6.SetTextFont(42)
leg6.SetTextSize(0.10)
leg6.AddEntry(g_loss_mean, "Generator loss", "l")
#leg6.AddEntry( g_loss_orig, "Original loss", "l" )
#leg6.AddEntry( g_loss_flip, "Flipped loss", "l" )
leg6.SetY1(leg1.GetY1() - 0.05 * leg1.GetNRows())
leg6.Draw()

gPad.RedrawAxis()

c.cd()

imgname = "img/training_%s.png" % dsid
c.SaveAs(imgname)
