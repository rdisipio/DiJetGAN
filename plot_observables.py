#!/usr/bin/env python

import os, sys

from ROOT import *
from math import sqrt, pow, log

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro( "AtlasUtils.C" )
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)

######## 

def Normalize( h, sf=1.0 ):
  if h == None: return

  A = h.Integral()
  if A == 0.: return

  h.Scale( sf / A )

#~~~~~~~~~

def DivideByBinWidth(h):
  nbins = h.GetNbinsX()
  for i in range(nbins):
     y  = h.GetBinContent(i+1)
     dy = h.GetBinError(i+1)
     bw = h.GetBinWidth(i+1)
     h.SetBinContent( i+1, y/bw )
     h.SetBinError( i+1, dy/bw )

#~~~~~~~~~

def SetTH1FStyle( h, color = kBlack, linewidth = 1, fillcolor = 0, fillstyle = 0, markerstyle = 21, markersize = 1.3, fill_alpha = 0 ):
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


def MakeCanvas( npads = 1, side = 800, split = 0.25, padding = 0.00 ):
    # assume that if pads=1 => square plot
    # padding used to be 0.05
    y_plot    = side * ( 1. - ( split + padding ) )
    y_ratio   = side * split
    y_padding = side * padding

    height_tot = y_plot + npads * ( y_ratio + y_padding )
    height_tot = int(height_tot)

    c = TCanvas( "PredictionData", "Prediction/Data", side, height_tot )
    c.SetFrameFillStyle(4000)
    c.SetFillColor(0) 

    pad0 = TPad( "pad0","pad0",0, split+padding,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0. )
    #pad0.SetTopMargin( 0.14 )
    pad0.SetTopMargin( 0.07 ) #0.05
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()

    pad1 = TPad( "pad1","pad1",0,0,1, split,0,0,0 )
    pad1.SetLeftMargin( 0.18 ) #0.16
    pad1.SetRightMargin( 0.05 )
    pad1.SetTopMargin( 0. )
#    pad1.SetBottomMargin( 0. )
    pad1.SetGridy(1)
    pad1.SetTopMargin(0)
    pad1.SetBottomMargin(0.5) #0.4
    pad1.Draw()
    pad1.SetFillColor(0)
    pad1.SetFillStyle(4000)

    pad0.cd()
    return c, pad0, pad1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MakeUncertaintyBand( prediction ):
    unc = TGraphAsymmErrors()
    
    i = 0

    if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
       Npoints = prediction.GetN()
    else:
       Npoints = prediction.GetNbinsX()

    for n in range( Npoints ):
       if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
          x_mc = Double()
          y_mc = Double()
          prediction.GetPoint( n, x_mc, y_mc )
       else:
          x_mc = prediction.GetBinCenter(n+1)
          y_mc = prediction.GetBinContent(n+1)

       if y_mc == 0: continue
    
       unc.SetPoint( i, x_mc, 1.0 )
      
       if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:  
          bw_l = prediction.GetErrorXlow( n )
          bw_h = prediction.GetErrorXhigh( n )
          err_y_lo = prediction.GetErrorYlow(n) / y_mc
          err_y_hi = prediction.GetErrorYhigh(n) / y_mc
       else:
          bw_l = prediction.GetBinWidth( n+1 ) / 2. 
          bw_h = prediction.GetBinWidth( n+1 ) / 2.
          err_y_lo = prediction.GetBinError( n+1 ) / y_mc
          err_y_hi = prediction.GetBinError( n+1 ) / y_mc

       unc.SetPointError( i, bw_l, bw_h, err_y_lo, err_y_hi )

       i += 1
  
    return unc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MakeRatio( data, prediction, setgr = True ):
    ratio = TGraphAsymmErrors()
    
    if setgr:
        SetTH1FStyle( ratio, color=data.GetMarkerColor(), markerstyle=data.GetMarkerStyle() )
    
    if data.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
       nbins = data.GetN()
    else:
       nbins = data.GetNbinsX()

    i = 0
    for n in range( nbins ):
        x_mc = Double()
        y_mc = Double()
        x_data = Double()
        y_data = Double()

        if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
           prediction.GetPoint( n, x_mc, y_mc )
        else:
           x_mc = prediction.GetBinCenter( n+1 )
           y_mc = prediction.GetBinContent( n+1 )   
     
        if y_mc == 0.: continue

        if data.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
           data.GetPoint( n, x_data, y_data )
           bw = data.GetErrorXlow(n) + data.GetErrorXhigh(n)
           dy_u = data.GetErrorYhigh(n)
           dy_d = data.GetErrorYlow(n)
        else:    
           x_data = data.GetBinCenter( n+1 )
           y_data = data.GetBinContent( n+1 )
           bw = data.GetBinWidth( n+1 )
           dy_u = data.GetBinError( n+1 )
           dy_d = data.GetBinError( n+1 ) 
        
        #print '    setting point %i: %f' % (i,y_data/y_mc,)

        ratio.SetPoint( i, x_data, y_data/y_mc )
        
        ratio.SetPointError( i, 0., 0., dy_d/y_mc, dy_u/y_mc )
        
        i += 1
    return ratio

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def DrawRatio( data, prediction, xtitle = "", yrange=[0.4,1.6] ):
    
    if data.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
       n = data.GetN()
       x = Double()
       y = Double()
       data.GetPoint( 0, x, y )
       exl = data.GetErrorXlow( 0 )
       xmin = x - exl
       data.GetPoint( n-1, x, y )
       exh = data.GetErrorXhigh( n-1 )
       xmax = x + exh
    else:
       xmin = data.GetXaxis().GetXmin()
       xmax = data.GetXaxis().GetXmax()

    # tt diffxs 7 TeV: [ 0.4, 1.6 ]    
#    frame = gPad.DrawFrame( xmin, 0.7, xmax, 1.3 )
    frame = gPad.DrawFrame( xmin, yrange[0], xmax, yrange[1] ) #2.1 
#    frame = gPad.DrawFrame( xmin, 0.3, xmax, 2.2 ) 
  
    frame.GetXaxis().SetNdivisions(508)
    frame.GetYaxis().SetNdivisions(504)
    
    frame.GetXaxis().SetLabelSize( 0.16 )
    frame.GetXaxis().SetTitleSize( 0.16 )
    frame.GetXaxis().SetTitleOffset( 1.2 )
    
    frame.GetYaxis().SetLabelSize( 0.16 )
    frame.GetYaxis().SetTitle( "#frac{Data}{Prediction}" )
    frame.GetYaxis().SetTitleSize( 0.16 )
    frame.GetYaxis().SetTitleOffset( 0.5 )
    
    frame.GetXaxis().SetTitle( xtitle )

    frame.Draw()
    
    tot_unc  = MakeUncertaintyBand( prediction )

    SetTH1FStyle( tot_unc,  color=kGray+1, fillstyle=1001, fillcolor=kGray+1, linewidth=0, markersize=0 )

    ratio   = MakeRatio( data, prediction, True )

    tot_unc.Draw( "e2 same" )
    ratio.Draw( "p same" )

    gPad.RedrawAxis()

    return frame, tot_unc, ratio

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

obs = "ljet1_pt"
dsid = "361024"
preselection = "incl"

if len(sys.argv) > 1: obs = sys.argv[1]
if len(sys.argv) > 2: dsid = sys.argv[2]
if len(sys.argv) > 3: preselection = sys.argv[3]

hname = "%s" % (obs)

infilename_GAN  = "histograms/histograms.mc16a.%s.GAN.%s.root" % ( dsid, preselection )
infilename_MC   = "histograms/histograms.mc16a.%s.MC.%s.root" % ( dsid, preselection )

infile_GAN = TFile.Open( infilename_GAN )
infile_MC  = TFile.Open( infilename_MC )

h_GAN = infile_GAN.Get( hname )
h_MC  = infile_MC.Get( hname )

#print hname, h_GAN, h_MC

xtitle = h_MC.GetXaxis().GetTitle()
ytitle = h_MC.GetYaxis().GetTitle()

if h_GAN.Class() == TH2F.Class():
#  print "INFO: calculating profile-X"
  h_GAN = h_GAN.ProfileX("GAN_pfx")
  h_MC  = h_MC.ProfileX("MC_pfx")
else:
  Normalize( h_GAN )
  Normalize( h_MC )

DivideByBinWidth( h_GAN )
DivideByBinWidth( h_MC )

SetTH1FStyle( h_GAN, color=kRed, markerstyle=24, markersize=2 )
SetTH1FStyle( h_MC,  color=kBlack, markerstyle=20, markersize=2)


#hmax = 1.3 * max( [h_GAN.GetMaximum(), h_MC.GetMaximum() ] )
hmax = 1.5 * h_MC.GetMaximum()
h_GAN.SetMaximum( hmax )
h_MC.SetMaximum( hmax )
h_GAN.SetMinimum( 0. )
h_MC.SetMinimum( 0. )

c, pad0, pad1 = MakeCanvas()

pad0.cd()

h_MC.Draw()
h_GAN.Draw("same")

leg = TLegend( 0.65, 0.90, 0.80, 0.90 )
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetTextSize(0.05)
leg.AddEntry( h_MC, "Real MC (%s)"%dsid, "lep" )
leg.AddEntry( h_GAN, "GAN MC", "lep" )
leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
leg.Draw()

l = TLatex()
l.SetNDC()
l.SetTextFont(72)
l.SetTextColor(kBlack)
l.DrawLatex(0.20, 0.85, "ATLAS Simulation Internal")

gPad.RedrawAxis()

pad1.cd()

yrange = [0.4, 1.6]
frame, tot_unc, ratio = DrawRatio( h_GAN, h_MC, xtitle, yrange )

gPad.RedrawAxis()

c.cd()

c.SaveAs( "img/GAN_%s_%s.png" % (dsid,hname) )
