from ROOT import *
from array import array
from math import pow, sqrt 
import numpy as np

GeV = 1e3
TeV = 1e6

def Normalize( h, sf=1.0, opt="width" ):
   area = h.Integral( opt )
   h.Scale( sf / area )

#####################

def GetEventWeight( tree, syst="nominal" ):
    w = 1.

    w_pileup   = tree.weight_pileup
    w_jvt      = tree.weight_jvt
    w_btag     = tree.weight_bTagSF_MV2c10_70
    w_others   = 1.
    isOtherSyst = True

    if syst == "nominal":
        pass

    elif syst in [ "pileup_UP", "pileup_DOWN" ]:
        w_pileup = tree.weight_pileup_UP if syst == "pileup_UP" else tree.weight_pileup_DOWN

    elif syst in [ "jvt_UP", "jvt_DOWN" ]:
        w_jvt = tree.weight_jvt_UP if syst == "jvt_UP" else tree.weight_jvt_DOWN

    elif syst.startswith("bTagSF"):
        if "eigenvars" in syst:
            syst_branch = "weight_%s" % syst
            exec( "w_btag = tree.%s" % syst_branch )
        else:
            k = int( syst.split('_')[-1] )
            syst_btag   = syst.replace( "_up_%i"%k, "_up" ).replace( "_down_%i"%k, "_down" )
            syst_branch = "weight_%s" % syst_btag
            exec( "w_btag = tree.%s[%i]" % (syst_branch, k ) )
    else:
        if syst in systematics_weight:
            syst_branch = "weight_%s" % syst
            exec( "w_others = tree.%s" % syst_branch )

    w *= w_pileup * w_jvt * w_btag * w_others

    return w

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MakeEventJets( tree, b_tag_cut=0.83 ):
    jets = []
    bjets = []
    ljets = []

    jets_n = len( tree.jet_pt )
    for i in range(jets_n):
        jets += [ TLorentzVector() ]
        jets[-1].SetPtEtaPhiE( tree.jet_pt[i], tree.jet_eta[i], tree.jet_phi[i], tree.jet_e[i] )
        jets[-1].index = i
        jets[-1].ntrk = max( 0, tree.jet_QGTaggerNTrack[i] )
 
        jets[-1].mv2c10 = tree.jet_mv2c10[i]
        jets[-1].isBtagged = False
        if jets[-1].mv2c10 > b_tag_cut:
            jets[-1].isBtagged = True
            bjets += [ TLorentzVector(jets[-1]) ]
            bjets[-1].mv2c10 = jets[-1].mv2c10
            bjets[-1].index = i
            bjets[-1].ntrk = jets[-1].ntrk

    ljets_n = len( tree.ljet_pt )
    for i in range(ljets_n):
       ljets += [ TLorentzVector() ]
       lj = ljets[-1]
       lj.SetPtEtaPhiM( tree.ljet_pt[i], tree.ljet_eta[i], tree.ljet_phi[i], tree.ljet_m[i] )
       lj.index = i
       lj.tau2  = tree.ljet_tau2[i]
       lj.tau3  = tree.ljet_tau3[i]
       lj.tau32 = tree.ljet_tau32[i]
       lj.tau21 = tree.ljet_tau21[i]
       lj.sd12  = tree.ljet_sd12[i]
       lj.sd23  = tree.ljet_sd23[i]
       lj.Qw    = tree.ljet_Qw[i]
       lj.ntrk  = tree.ljet_QGTaggerNTrack[i]

       lj.bmatch_dR_min = 10
       lj.bmatch_n = 0
       for bj in bjets:
         dR = lj.DeltaR(bj)
         if dR > 1.2: continue
         if dR > lj.bmatch_dR_min: continue
         lj.bmatch_dR_min = dR
         lj.bmatch_n += 1

    return jets, bjets, ljets


###############################

xedges = {
#  'tt_m'	  : array( 'd', [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.1, 2.3, 3.0 ] ),
  'tt_m'         : array( 'd', [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.25, 2.5, 3.0, 3.5 ] ),
  'tt_pt'     : array( 'd', [ 0, 100, 150, 200, 300, 400, 500, 600, 800 ] ),
  'tt_y'	  : array( 'd', [ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 ] ),
  'tt_cosThS'   : array( 'd', [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ] ),
  'tt_Pout'   : array( 'd', [ 0., 20., 40., 60., 80, 100., 120, 150, 200., 250, 300., 350, 450, 600. ] ),
  'tt_chi'    : array( 'd', [ 1., 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30 ] ),
}

htitle = {
  't1_pt'   : ";p_{T}^{t,1} [GeV]",
  't1_y'    : ";|y^{t,1}|",

  't2_pt'   : ";p_{T}^{t,2} [GeV]",
  't2_y'    : ";|y^{t,2}|",

  'tt_m'    : ";m^{t#bar{t}} [TeV]",
  'tt_pt'   : ";p_{T}^{t#bar{t}} [GeV]",
  'tt_y'    : ";y^{t#bar{t}}",
  'tt_Pout' : ";|P_{out}^{t#bar{t}}| [GeV]",
  'tt_cosThS' : ";|cos #theta^{*}|",  
  'tt_z'    : ";p_{T}^{t,2} / p_{T}^{t,1}",
  'tt_dR'   : ";#Delta R(t, #bar{t})", 
  'tt_dPhi' : ";#Delta Phi(t, #bar{t})", 
  'tt_chi'  : ";#chi^{t#bar{t}}",
}

# ~~~~~~~~~~~~~~~~~~~~~~

def BookHistograms( histograms, ctag ):
  histograms['%s_DNN'         % ctag ] = TH1F( "%s_DNN"         % ctag, ";Classifier output", 40, 0., 1. )

  histograms['%s_inclusive'   % ctag ] = TH1F( "%s_inclusive"   % ctag, ";Total number of events", 1, 0.5, 1.5 )

  histograms['%s_jets_n'      % ctag ] = TH1F( "%s_jets_n"      % ctag, ";Anti-k_{T} R=0.4 jets multiplicity", 12, -0.5, 11.5 )
  histograms['%s_bjets_n'     % ctag ] = TH1F( "%s_bjets_n"     % ctag, ";Anti-k_{T} R=0.4 b-tagged jets multiplicity", 5, -0.5, 4.5 )
  histograms['%s_ljets_n'     % ctag ] = TH1F( "%s_ljets_n"     % ctag, ";Anti-k_{T} R=1.0 jets multiplicity", 5, -0.5, 4.5 )  

  histograms['%s_ljet1_m'     % ctag ] = TH1F( "%s_ljet1_m"     % ctag, ";Leading large-R jet mass [GeV]", 20, 120, 220 )
  histograms['%s_ljet1_pt'    % ctag ] = TH1F( "%s_ljet1_pt"    % ctag, ";Leading large-R jet p_{T} [GeV]", 25, 500, 1500 )
  histograms['%s_ljet1_tau32' % ctag ] = TH1F( "%s_ljet1_tau32" % ctag, ";Leading large-R jet #tau_{32}", 20, 0., 1. )
  histograms['%s_ljet1_sd12'  % ctag ] = TH1F( "%s_ljet1_sd12"  % ctag, ";Leading large-R jet #sqrt{d_{12}}", 20, 0., 200. )
  histograms['%s_ljet1_sj1_m' % ctag ] = TH1F( "%s_ljet1_sj1_m" % ctag, ";Leading subjet mass [GeV]", 30, 0., 120. )
  histograms['%s_ljet1_dRb_min'  % ctag ] = TH1F( "%s_ljet1_dRb_min"  % ctag, ";min #Delta R(J_{1}, b)", 20, 0., 1. )
  histograms['%s_ljet1_tau2'  % ctag ] = TH1F( "%s_ljet1_tau2"  % ctag, ";Leading large-R jet #tau_{2}", 15, 0., 0.30 )
  histograms['%s_ljet1_tau3'  % ctag ] = TH1F( "%s_ljet1_tau3"  % ctag, ";Leading large-R jet #tau_{3}", 15, 0., 0.15 )
  histograms['%s_ljet1_sj_n'  % ctag ] = TH1F( "%s_ljet1_sj_n"  % ctag, ";Matched jets multiplicity", 5, 0.5, 5.5 )

  histograms['%s_ljet2_m'     % ctag ] = TH1F( "%s_ljet2_m"     % ctag, ";Subleading large-R jet mass [GeV]", 20, 120, 220 )
  histograms['%s_ljet2_pt'    % ctag ] = TH1F( "%s_ljet2_pt"    % ctag, ";Subleading large-R jet p_{T} [GeV]", 30, 300, 1500 )
  histograms['%s_ljet2_tau32' % ctag ] = TH1F( "%s_ljet2_tau32" % ctag, ";Subleading large-R jet #tau_{32}", 20, 0., 1. )
  histograms['%s_ljet2_sd12'  % ctag ] = TH1F( "%s_ljet2_sd12"  % ctag, ";Subeading large-R jet #sqrt{d_{12}}", 20, 0., 200. )
  histograms['%s_ljet2_sj1_m' % ctag ] = TH1F( "%s_ljet2_sj1_m" % ctag, ";Leading subjet mass [GeV]", 30, 0., 120. )
  histograms['%s_ljet2_dRb_min'  % ctag ] = TH1F( "%s_ljet2_dRb_min"  % ctag, ";min #Delta R(J_{2}, b)", 20, 0., 1. )
  histograms['%s_ljet2_tau2'  % ctag ] = TH1F( "%s_ljet2_tau2"  % ctag, ";Leading large-R jet #tau_{2}", 15, 0., 0.30 )
  histograms['%s_ljet2_tau3'  % ctag ] = TH1F( "%s_ljet2_tau3"  % ctag, ";Leading large-R jet #tau_{3}", 15, 0., 0.15 )
  histograms['%s_ljet2_sj_n'  % ctag ] = TH1F( "%s_ljet2_sj_n"  % ctag, ";Matched jets multiplicity", 5, 0.5, 5.5 )

  histograms['%s_jet1_pt'      % ctag ] = TH1F( "%s_jet1_pt"      % ctag, ";1st anti-k_{T} R=0.4 jet p_{T} [GeV]", 15, 0., 1500 )
  histograms['%s_jet2_pt'      % ctag ] = TH1F( "%s_jet2_pt"      % ctag, ";2nd anti-k_{T} R=0.4 jet p_{T} [GeV]", 15, 0., 1500 )
  histograms['%s_jet3_pt'      % ctag ] = TH1F( "%s_jet3_pt"      % ctag, ";3rd anti-k_{T} R=0.4 jet p_{T} [GeV]", 15, 0., 500 )
  histograms['%s_jet4_pt'      % ctag ] = TH1F( "%s_jet4_pt"      % ctag, ";4th anti-k_{T} R=0.4 jet p_{T} [GeV]", 15, 0., 300 )

#  histograms['%s_bjet1_pt'      % ctag ] = TH1F( "%s_bjet1_pt"    % ctag, ";1st b-jet p_{T} [GeV]", 20, 0., 1000 )
#  histograms['%s_bjet2_pt'      % ctag ] = TH1F( "%s_bjet2_pt"    % ctag, ";2nd b-jet p_{T} [GeV]", 15, 0., 1000 )
#  histograms['%s_bjet3_pt'      % ctag ] = TH1F( "%s_bjet3_pt"    % ctag, ";3rd b-jet p_{T} [GeV]", 15, 0., 500 )

#  histograms['%s_bjet1_mv2c10' % ctag ] = TH1F( "%s_bjet1_mv2c10" % ctag, ";Leading b-jet weight",     20, 0., 1. )
#  histograms['%s_bjet2_mv2c10' % ctag ] = TH1F( "%s_bjet2_mv2c10" % ctag, ";2nd-leading b-jet weight", 20, 0., 1. )
#  histograms['%s_bjet3_mv2c10' % ctag ] = TH1F( "%s_bjet3_mv2c10" % ctag, ";3rd-leading b-jet weight", 20, 0., 1. )

  histograms['%s_jet1_mv2c10'     % ctag ] = TH1F( "%s_jet1_mv2c10"     % ctag, ";1st anti-k_{T} R=0.4 jet b-tag weight", 20, 0., 1. )
  histograms['%s_jet2_mv2c10'     % ctag ] = TH1F( "%s_jet2_mv2c10"     % ctag, ";2nd anti-k_{T} R=0.4 jet b-tag weight", 20, 0., 1. )
  histograms['%s_jet3_mv2c10'     % ctag ] = TH1F( "%s_jet3_mv2c10"     % ctag, ";3rd anti-k_{T} R=0.4 jet b-tag weight", 20, 0., 1. )
  histograms['%s_jet4_mv2c10'     % ctag ] = TH1F( "%s_jet4_mv2c10"     % ctag, ";4th anti-k_{T} R=0.4 jet b-tag weight", 20, 0., 1. )

  histograms['%s_jet1_Ntrk'     % ctag ] = TH1F( "%s_jet1_Ntrk"         % ctag, ";1st anti-k_{T} R=0.4 jet N_{trk}", 40, 0.5, 40.5 )
  histograms['%s_jet2_Ntrk'     % ctag ] = TH1F( "%s_jet2_Ntrk"	       	% ctag, ";2nd anti-k_{T} R=0.4 jet N_{trk}", 40, 0.5, 40.5 )
  histograms['%s_jet3_Ntrk'     % ctag ] = TH1F( "%s_jet3_Ntrk"	       	% ctag, ";3rd anti-k_{T} R=0.4 jet N_{trk}", 30, 0.5, 30.5 )
  histograms['%s_jet4_Ntrk'     % ctag ] = TH1F( "%s_jet4_Ntrk"	       	% ctag, ";4th anti-k_{T} R=0.4 jet N_{trk}", 30, 0.5, 30.5 )

  histograms['%s_tt_m' % ctag ]      = TH1F( "%s_tt_m"      % ctag, htitle['tt_m'],  len(xedges['tt_m'])-1, xedges['tt_m'] )
  histograms['%s_tt_pt' % ctag ]     = TH1F( "%s_tt_pt"     % ctag, htitle['tt_pt'], len(xedges['tt_pt'])-1, xedges['tt_pt'] )
  histograms['%s_tt_y' % ctag ]      = TH1F( "%s_tt_y"      % ctag, htitle['tt_y'],  len(xedges['tt_y'])-1, xedges['tt_y'] )
  histograms['%s_tt_z' % ctag ]      = TH1F( "%s_tt_z"      % ctag, htitle['tt_z'], 20, 0.5, 1.0 )
  histograms['%s_tt_Pout' % ctag ]   = TH1F( "%s_tt_Pout"   % ctag, htitle['tt_Pout'], len(xedges['tt_Pout'])-1, xedges['tt_Pout'] )
  histograms['%s_tt_cosThS' % ctag ] = TH1F( "%s_tt_cosThS" % ctag, htitle['tt_cosThS'], len(xedges['tt_cosThS'])-1, xedges['tt_cosThS'] )
  histograms['%s_tt_chi' % ctag ] = TH1F( "%s_tt_chi"       % ctag, htitle['tt_chi'], len(xedges['tt_chi'])-1, xedges['tt_chi'] )

  return histograms


#~~~~~~~~~~~~~~~~~~~~~~~~


def FillHistograms( histograms, event, ctag="ALL", w=1.0 ):

   histograms['%s_inclusive'%ctag].Fill( 1., w )
   histograms['%s_DNN'%ctag].Fill( event['dnn'], w )

   histograms['%s_jets_n'%ctag].Fill(      event['jets_n'], w )
   histograms['%s_bjets_n'%ctag].Fill(     event['bjets_n'], w )
   histograms['%s_ljets_n'%ctag].Fill(     event['ljets_n'], w )

   if event['jets_n'] > 0: histograms['%s_jet1_pt'%ctag].Fill( event['jet1_pt'] / GeV, w )
   if event['jets_n'] > 1: histograms['%s_jet2_pt'%ctag].Fill( event['jet2_pt'] / GeV, w )
   if event['jets_n'] > 2: histograms['%s_jet3_pt'%ctag].Fill( event['jet3_pt'] / GeV, w )
   if event['jets_n'] > 3: histograms['%s_jet4_pt'%ctag].Fill( event['jet4_pt'] / GeV, w )

   if event['jets_n'] > 0: histograms['%s_jet1_mv2c10'%ctag].Fill( event['jet1_mv2c10'], w )
   if event['jets_n'] > 1: histograms['%s_jet2_mv2c10'%ctag].Fill( event['jet2_mv2c10'], w )
   if event['jets_n'] > 2: histograms['%s_jet3_mv2c10'%ctag].Fill( event['jet3_mv2c10'], w )
   if event['jets_n'] > 3: histograms['%s_jet4_mv2c10'%ctag].Fill( event['jet4_mv2c10'], w )

   if event['jets_n'] > 0: histograms['%s_jet1_Ntrk'%ctag].Fill( event['jet1_Ntrk'], w )
   if event['jets_n'] > 1: histograms['%s_jet2_Ntrk'%ctag].Fill( event['jet2_Ntrk'], w )
   if event['jets_n'] > 2: histograms['%s_jet3_Ntrk'%ctag].Fill( event['jet3_Ntrk'], w )
   if event['jets_n'] > 3: histograms['%s_jet4_Ntrk'%ctag].Fill( event['jet4_Ntrk'], w )

   histograms['%s_ljet1_m'%ctag].Fill(     event['ljet1_m'], w )
   histograms['%s_ljet1_pt'%ctag].Fill(    event['ljet1_pt'], w )
   histograms['%s_ljet1_tau32'%ctag].Fill( event['ljet1_tau32'], w )
   histograms['%s_ljet1_sd12'%ctag].Fill(  event['ljet1_sd12'], w )
   histograms['%s_ljet1_sj1_m'%ctag].Fill( event['ljet1_sj1_m'], w )
   histograms['%s_ljet1_dRb_min'%ctag].Fill(  event['ljet1_dRb_min'], w )
   histograms['%s_ljet1_tau2'%ctag].Fill(  event['ljet1_tau2'], w )
   histograms['%s_ljet1_tau3'%ctag].Fill(  event['ljet1_tau3'], w )
   histograms['%s_ljet1_sj_n'%ctag].Fill(  event['ljet1_sj_n'], w )
#   for dR in event['ljet1_dRb: histograms['%s_ljet1_dRb'%ctag].Fill( dR, w )

   histograms['%s_ljet2_m'%ctag].Fill(     event['ljet2_m'], w )
   histograms['%s_ljet2_pt'%ctag].Fill(    event['ljet2_pt'], w )
   histograms['%s_ljet2_tau32'%ctag].Fill( event['ljet2_tau32'], w )
   histograms['%s_ljet2_sd12'%ctag].Fill(  event['ljet2_sd12'], w )
   histograms['%s_ljet2_sj1_m'%ctag].Fill( event['ljet2_sj1_m'], w )
   histograms['%s_ljet2_dRb_min'%ctag].Fill(  event['ljet2_dRb_min'], w )
   histograms['%s_ljet2_tau2'%ctag].Fill(  event['ljet2_tau2'], w )
   histograms['%s_ljet2_tau3'%ctag].Fill(  event['ljet2_tau3'], w )
   histograms['%s_ljet2_sj_n'%ctag].Fill(  event['ljet2_sj_n'], w )
#   for dR in event['ljet2_dRb: histograms['%s_ljet2_dRb'%ctag].Fill( dR, w )

   histograms['%s_tt_m'%ctag].Fill( event['tt_m'], w )
   histograms['%s_tt_pt'%ctag].Fill( event['tt_pt'], w )
   histograms['%s_tt_y'%ctag].Fill( abs(event['tt_y']), w )
   histograms['%s_tt_cosThS'%ctag].Fill( abs(event['tt_cosThS']), w )
   histograms['%s_tt_z'%ctag].Fill( event['tt_z'], w )
   histograms['%s_tt_Pout'%ctag].Fill( abs(event['tt_Pout']), w )
   histograms['%s_tt_chi'%ctag].Fill( abs(event['tt_chi']), w )

##############

def RotateEvent( jets, bjets, ljets = [], phi=0., flip_eta=False ):
    jets_new = []
    for j in jets:
        jets_new += [ TLorentzVector(j) ]
        j_new = jets_new[-1]

        j_new.ntrk   = j.ntrk
        j_new.mv2c10 = j.mv2c10
        j_new.index  = j.index

        j_new.RotateZ( phi )
        if flip_eta:
           j_new.SetPtEtaPhiE( j_new.Pt(), -j_new.Eta(), j_new.Phi(), j_new.E() )

    bjets_new = []
    for bj in bjets:
       bjets_new += [ TLorentzVector(bj) ]
       bj_new = bjets_new[-1]

       bj_new.ntrk   = bj.ntrk
       bj_new.mv2c10 = bj.mv2c10
       bj_new.index  = bj.index

       bj_new.RotateZ( phi )
       if flip_eta:
          bj_new.SetPtEtaPhiE( bj_new.Pt(), -bj_new.Eta(), bj_new.Phi(), bj_new.E() )

    ljets_new = []
    for lj in ljets:
       ljets_new += [ TLorentzVector(lj) ]
       lj_new = ljets_new[-1]
       lj_new.index  = lj.index

       lj_new.tau2   = lj.tau2
       lj_new.tau3   = lj.tau3
       lj_new.tau32  = lj.tau32
       lj_new.sd12   = lj.sd12
       lj_new.sd23   = lj.sd23
       lj_new.Qw     = lj.Qw
       lj_new.ntrk   = lj.ntrk
       lj_new.bmatch_dR_min = lj.bmatch_dR_min
       lj_new.bmatch_n      = lj.bmatch_n

       lj_new.RotateZ( phi )
       if flip_eta:
          lj_new.SetPtEtaPhiM( lj_new.Pt(), -lj_new.Eta(), lj_new.Phi(), lj_new.M() )

    return jets_new, bjets_new, ljets_new

####################


def make_rnn_input( jets, do_linearize=True ):
    event = np.zeros( ( n_fso_max, n_features_per_fso ) )

    jets_n = len(jets)
    jets_n_max = min( n_fso_max, jets_n )

    for i in range( jets_n_max ):
        event[i][0] = jets[i].Px() / GeV
        event[i][1] = jets[i].Py() / GeV
        event[i][2] = jets[i].Pz() / GeV
        event[i][3] = jets[i].M( ) / GeV
        event[i][4] = jets[i].mv2c10
        event[i][5] = jets[i].ntrk

    # linearize: sequence of ops is: scale->reshape->classify
    if do_linearize == True:
      event = event.reshape( ( n_features ) )

    return event

#~~~~~~~~~~~~~~~~~~~~

def make_rnn_input_highlevel( ljets, shape, do_linearize=True ):

    n_fso_max          = shape[0]
    n_features_per_fso = shape[1]
    n_features         = n_fso_max * n_features_per_fso

    event = np.zeros( ( n_fso_max, n_features_per_fso ) )

    ljets_n = len(ljets)
    ljets_n_max = min( n_fso_max, ljets_n )

    for i in range( ljets_n_max ):
        event[i][0]  = ljets[i].Px() / GeV
        event[i][1]  = ljets[i].Py() / GeV
        event[i][2]  = ljets[i].Pz() / GeV
        event[i][3]  = ljets[i].E() / GeV
        event[i][4]  = ljets[i].tau2
        event[i][5]  = ljets[i].tau3
        event[i][6]  = ljets[i].sd12 / GeV
        event[i][7]  = ljets[i].sd23 / GeV
        event[i][8]  = ljets[i].Qw / GeV
        event[i][9]  = ljets[i].bmatch_dR_min
        event[i][10] = ljets[i].bmatch_n
        event[i][11] = ljets[i].ntrk


    # linearize: sequence of ops is: scale->reshape->classify
    if do_linearize == True:
      event = event.reshape( ( n_features ) )

    return event


 #~~~~~~~~~~~~~~~~~~~~

def make_rnn_input_GAN( ljets, shape, do_linearize=True ):

    n_fso_max          = shape[0]
    n_features_per_fso = shape[1]
    n_features         = n_fso_max * n_features_per_fso

    event = np.zeros( ( n_fso_max, n_features_per_fso ) )

    ljets_n = len(ljets)
    ljets_n_max = min( n_fso_max, ljets_n )

    for i in range( ljets_n_max ):
        event[i][0]  = ljets[i].Px() / GeV
        event[i][1]  = ljets[i].Py() / GeV
        event[i][2]  = ljets[i].Pz() / GeV
        event[i][3]  = ljets[i].Pt() / GeV
        event[i][4]  = ljets[i].Eta()
        event[i][5]  = ljets[i].Phi()
        event[i][6]  = ljets[i].E() / GeV
        event[i][7]  = ljets[i].M() / GeV
        event[i][8]  = ljets[i].tau2
        event[i][9]  = ljets[i].tau3
        event[i][10]  = ljets[i].tau32
        #event[i][6]  = ljets[i].sd12 / GeV
        #event[i][7]  = ljets[i].sd23 / GeV
        #event[i][8]  = ljets[i].Qw / GeV
        #event[i][9]  = ljets[i].bmatch_dR_min
        #event[i][10] = ljets[i].bmatch_n
        #event[i][11] = ljets[i].ntrk


    # linearize: sequence of ops is: scale->reshape->classify
    if do_linearize == True:
      event = event.reshape( ( n_features ) )

    return event
 
####################################

def Interpolate( x, xvalues, yvalues ):
   if x < xvalues[0]:  return yvalues[0]
   if x > xvalues[-1]: return yvalues[-1]

   i = 0
   while xvalues[i+1] < x: i += 1
#   i = sum([ int(i<x) for i in xvalues ]) - 1
   xlow = xvalues[i]
   xup  = xvalues[i+1]
   a    = ( yvalues[i+1] - yvalues[i] ) / ( xup - xlow )
   b    = yvalues[i] - a*xlow
   y    = a*x + b
#   print x, i, xlow, xup, '-->', y

   return y
#   return yvalues[i] + ( yvalues[i+1] - yvalues[i] ) * ( x - xlow ) / ( xup - xlow )


#~~~~~~~~~~~~~~~~~~~~~~~
 

def TopSubstructureTagger( jet, wp="50", cut="full" ):
   pt_bins = array( 'd', [ 250000.000,325000.000,375000.000,425000.000,475000.000,525000.000,575000.000,625000.000,675000.000,725000.000,775000.000,850000.000,950000.000,1100000.000,1300000.000,1680000.000 ] )

   if wp == "50":
#     print "DEBUG: tight tagging"
     tau32_cuts = array( 'd', [ 0.773,0.713,0.672,0.637,0.610,0.591,0.579,0.574,0.573,0.574,0.576,0.578,0.580,0.580,0.577,0.571 ] )
     mass_cuts  = array( 'd', [ 85052.983,98775.422,107807.048,115186.721,120365.410,123510.000,125010.000,125662.377,126075.960,126389.113,126537.840,126803.137,127322.903,128379.386,130241.032,133778.159 ] )
   elif wp == "80":
#     print "DEBUG: loose tagging"
     tau32_cuts = array( 'd', [ 0.879,0.831,0.799,0.777,0.746,0.727,0.714,0.776,0.771,0.698,0.698,0.699,0.770,0.771,0.699,0.696 ] )
     mass_cuts  = array( 'd', [ 67888.967,72014.026,74764.066,76769.667,78354.344,79177.000,79530.000,80158.525,81195.851,82779.245,84890.965,88747.162,94262.629,102710.787,113868.253,135067.438 ] )
   else:
     print "ERROR: TopSubstructureTagger: unknown working point", wp

   jet_pt    = jet.Pt()
   jet_m     = jet.M()
   jet_tau32 = jet.tau32

   cut_m     = Interpolate( jet_pt, pt_bins, mass_cuts )
   cut_tau32 = Interpolate( jet_pt, pt_bins, tau32_cuts )
   pass_mass_cut  = jet_m > cut_m
   pass_tau32_cut = jet_tau32 < cut_tau32

#   print "DEBUG: wp=%s : pT=%4.1f m=%4.1f tau32=%4.3f : cut_m=%4.1f cut_tau32=%4.3f : pass_mass_cut=%i pass_tau32_cut=%i" % ( wp, jet_pt/GeV, jet_m/GeV, jet_tau32, cut_m/GeV, cut_tau32, pass_mass_cut, pass_tau32_cut )
#   return pass_tau32_cut
   if cut == "full":
     return ( pass_mass_cut and pass_tau32_cut )
   elif cut == "tau32":
     return pass_tau32_cut
   elif cut == "mass":
     return pass_mass_cut
   else:
     return None



##########################################

