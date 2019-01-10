#!/usr/bin/env python

import os, sys

from ROOT import *
from math import sqrt, pow, log

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro( "AtlasUtils.C" )
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def SetHistogramStyle( h, color = kBlack, linewidth = 1, fillcolor = 0, fillstyle = 0, markerstyle = 21, markersize = 1.3, fill_alpha = 0 ):
    '''Set the style with a long list of parameters'''

    h.SetLineColor( color )
    h.SetLineWidth( linewidth )
    h.SetFillColor( fillcolor )
    h.SetFillStyle( fillstyle )
    h.SetMarkerStyle( markerstyle )
    h.SetMarkerColor( h.GetLineColor() )
    h.SetMarkerSize( markersize )
    if fill_alpha > 0:
       h.SetFillColorAlpha( color, fill_alpha )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dsid = "361024"
preselection = "incl"

if len(sys.argv) > 1: dsid = sys.argv[1]
    
infilename = "GAN/training_history.%s.rnn.GAN.%s.nominal.root" % ( dsid, preselection )
infile = TFile.Open( infilename )

d_loss = infile.Get( "d_loss" )
d_loss_r = infile.Get( "d_loss_r" )
d_loss_f = infile.Get( "d_loss_f") 
g_loss = infile.Get( "g_loss" )

n_epochs = d_loss.GetN()
frame_d_loss = TH1F( "frame_d_loss", ";Training Epoch;Discriminator Loss", 10, 0, n_epochs )
frame_g_loss = TH1F( "frame_g_loss", ";Training Epoch;Generator Loss", 10, 0, n_epochs )

hmax = 1.3
frame_d_loss.SetMaximum( hmax )
frame_d_loss.SetMinimum( 0. )

hmax = 1.3*TMath.MaxElement( g_loss.GetN(), g_loss.GetY() )
frame_g_loss.SetMaximum( hmax )
frame_g_loss.SetMinimum( 0. )

SetHistogramStyle( d_loss_f, color=kRed )
SetHistogramStyle( d_loss_r, color=kBlue )

c = TCanvas( "C", "C", 800, 1200 )
c.Divide( 1, 2 )

c.cd(1)
frame_d_loss.Draw()
d_loss_f.Draw("l same")
d_loss_r.Draw("l same")
d_loss.Draw("l same" )

leg1 = TLegend( 0.65, 0.90, 0.80, 0.90 )
leg1.SetFillColor(0)
leg1.SetFillStyle(0)
leg1.SetBorderSize(0)
leg1.SetTextFont(42)
leg1.SetTextSize(0.05)
leg1.AddEntry( d_loss,   "Average loss", "l" )
leg1.AddEntry( d_loss_r, "Real loss", "l" )
leg1.AddEntry( d_loss_f, "Fake loss", "l" )
leg1.SetY1( leg1.GetY1() - 0.05 * leg1.GetNRows() )
leg1.Draw()

gPad.RedrawAxis()

c.cd(2)
frame_g_loss.Draw()
g_loss.Draw("l same" )

c.cd()

imgname = "img/training_%s.png" % dsid
c.SaveAs( imgname )
