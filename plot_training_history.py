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


level = "ptcl"
preselection = "pt250"
dsid = "mg5_dijet_ht500"

if len(sys.argv) > 1:
    level = sys.argv[1]
if len(sys.argv) > 2:
    preselection = sys.argv[2]
if len(sys.argv) > 3:
    dsid = sys.argv[3]

if not os.path.exists("img/%s/"%level):
    os.makedirs("img/%s/"%level)

infilename = "GAN_%s/DCGAN.training_history.%s.%s.%s.nominal.root" % (
    level, dsid, level, preselection)
infile = TFile.Open(infilename)

d_loss = infile.Get("d_loss")
d_loss_r = infile.Get("d_loss_r")
d_loss_f = infile.Get("d_loss_f")
g_loss = infile.Get("g_loss")
d_acc = infile.Get("d_acc")
d_acc_f = infile.Get("d_acc_f")
d_acc_r = infile.Get("d_acc_r")

n_epochs = d_loss.GetN()
frame_loss = TH1F(
    "frame_loss", ";Loss", 10, 0, n_epochs)
frame_d_loss = TH1F(
    "frame_d_loss", ";Training Epoch;Loss", 10, 0, n_epochs)
frame_g_loss = TH1F(
    "frame_g_loss", ";Training Epoch;Loss", 10, 0, n_epochs)
frame_d_acc = TH1F(
    "frame_d_acc",  ";Training Epoch;Discriminator Accuracy", 10, 0, n_epochs)

#hmax = 1.6
hmax = 2.5
frame_d_loss.SetMaximum(hmax)
frame_d_loss.SetMinimum(0)
frame_d_loss.GetXaxis().SetTitleSize(0.07)
frame_d_loss.GetYaxis().SetTitleSize(0.07)
frame_d_loss.GetXaxis().SetLabelSize(0.07)
frame_d_loss.GetYaxis().SetLabelSize(0.07)
frame_d_loss.GetYaxis().SetTitleOffset(0.7)
frame_d_loss.GetXaxis().SetTitleOffset(1.0)

hmax = 1.6
frame_d_acc.SetMaximum(hmax)
frame_d_acc.SetMinimum(0)
frame_d_acc.GetXaxis().SetTitleSize(0.07)
frame_d_acc.GetYaxis().SetTitleSize(0.07)
frame_d_acc.GetXaxis().SetLabelSize(0.07)
frame_d_acc.GetYaxis().SetLabelSize(0.07)
frame_d_acc.GetYaxis().SetTitleOffset(0.7)
frame_d_acc.GetXaxis().SetTitleOffset(1.0)

hmax = 1.6
frame_g_loss.SetMaximum(hmax)
frame_g_loss.SetMinimum(0)
frame_g_loss.GetXaxis().SetTitleSize(0.07)
frame_g_loss.GetYaxis().SetTitleSize(0.07)
frame_g_loss.GetXaxis().SetLabelSize(0.07)
frame_g_loss.GetYaxis().SetLabelSize(0.07)
frame_g_loss.GetYaxis().SetTitleOffset(0.7)

SetHistogramStyle(d_loss_f, color=kRed, linewidth=3)
SetHistogramStyle(d_loss_r, color=kBlue, linewidth=3)
SetHistogramStyle(d_loss,   color=kOrange-3, linewidth=3)

SetHistogramStyle(d_acc_f, color=kRed, linewidth=3)
SetHistogramStyle(d_acc_r, color=kBlue, linewidth=3)

SetHistogramStyle(g_loss, color=kGreen-1, linewidth=3)

c = TCanvas("C", "C", 1600, 1200)

gPad.Divide(1, 2)

# Loss (orig)
c.cd(1)
gPad.SetLeftMargin(0.10)
gPad.SetTopMargin(0.05)
gPad.SetBottomMargin(0.15)
gPad.SetRightMargin(0.07)

frame_d_loss.Draw()
g_loss.Draw("l_same")
#d_loss_f.Draw("l same")
#d_loss_r.Draw("l same")
d_loss.Draw("l same")

leg1 = TLegend(0.65, 0.90, 0.90, 0.90)
leg1.SetNColumns(1)
leg1.SetFillColor(0)
leg1.SetFillStyle(0)
leg1.SetBorderSize(0)
leg1.SetTextFont(42)
leg1.SetTextSize(0.07)
leg1.AddEntry(d_loss,   "Discriminator loss", "l")
#leg1.AddEntry(d_loss_r, "D loss (r)", "l")
#leg1.AddEntry(d_loss_f, "D loss (f)", "l")
leg1.AddEntry(g_loss,   "Generator loss", "l")
leg1.SetY1(leg1.GetY1() - 0.10 * leg1.GetNRows())
leg1.Draw()

gPad.RedrawAxis()

# Acc (orig)
c.cd(2)
gPad.SetLeftMargin(0.10)
gPad.SetTopMargin(0.05)
gPad.SetBottomMargin(0.15)
gPad.SetRightMargin(0.07)

frame_d_acc.Draw()
d_acc_f.Draw("l same")
d_acc_r.Draw("l same")
d_acc.Draw("l same")

leg2 = TLegend(0.20, 0.90, 0.80, 0.90)
leg2.SetNColumns(3)
leg2.SetFillColor(0)
leg2.SetFillStyle(0)
leg2.SetBorderSize(0)
leg2.SetTextFont(42)
leg2.SetTextSize(0.07)
leg2.AddEntry(d_acc,   "Average acc", "l")
leg2.AddEntry(d_acc_r, "Real acc", "l")
leg2.AddEntry(d_acc_f, "Fake acc", "l")
leg2.SetY1(leg1.GetY1() - 0.05 * leg1.GetNRows())
leg2.Draw()

gPad.RedrawAxis()

# Generator loss
# c.cd(3)
# frame_g_loss.Draw()

#g_loss.Draw("l same")

#leg6 = TLegend(0.65, 0.90, 0.80, 0.90)
# leg6.SetFillColor(0)
# leg6.SetFillStyle(0)
# leg6.SetBorderSize(0)
# leg6.SetTextFont(42)
# leg6.SetTextSize(0.07)
#leg6.AddEntry(g_loss, "Generator loss", "l")
#leg6.SetY1(leg1.GetY1() - 0.05 * leg1.GetNRows())
# leg6.Draw()

# gPad.RedrawAxis()

c.cd()

imgname = "img/%s/training_%s_%s_%s.png" % (level,dsid, level, preselection)
c.SaveAs(imgname)
