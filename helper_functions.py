from ROOT import *
from array import array
from math import pow, sqrt
import numpy as np

GeV = 1e3
TeV = 1e6

#####################

def Normalize(h, sf=1.0, opt="width"):
    area = h.Integral(opt)
    h.Scale(sf / area)

#####################


def GetEventWeight(tree, syst="nominal"):
    w = 1.

    w_pileup = tree.weight_pileup
    w_jvt = tree.weight_jvt
    w_btag = tree.weight_bTagSF_MV2c10_70
    w_others = 1.
    isOtherSyst = True

    if syst == "nominal":
        pass

    elif syst in ["pileup_UP", "pileup_DOWN"]:
        w_pileup = tree.weight_pileup_UP if syst == "pileup_UP" else tree.weight_pileup_DOWN

    elif syst in ["jvt_UP", "jvt_DOWN"]:
        w_jvt = tree.weight_jvt_UP if syst == "jvt_UP" else tree.weight_jvt_DOWN

    elif syst.startswith("bTagSF"):
        if "eigenvars" in syst:
            syst_branch = "weight_%s" % syst
            exec("w_btag = tree.%s" % syst_branch)
        else:
            k = int(syst.split('_')[-1])
            syst_btag = syst.replace("_up_%i" % k, "_up").replace(
                "_down_%i" % k, "_down")
            syst_branch = "weight_%s" % syst_btag
            exec("w_btag = tree.%s[%i]" % (syst_branch, k))
    else:
        if syst in systematics_weight:
            syst_branch = "weight_%s" % syst
            exec("w_others = tree.%s" % syst_branch)

    w *= w_pileup * w_jvt * w_btag * w_others

    return w

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def MakeEventJets(tree, b_tag_cut=0.83):
    ljets = []

    ljets_n = len(tree.ljet_pt)
    for i in range(ljets_n):
        ljets += [TLorentzVector()]
        lj = ljets[-1]
        lj.SetPtEtaPhiM(
            tree.ljet_pt[i]/GeV, tree.ljet_eta[i], tree.ljet_phi[i], tree.ljet_m[i]/GeV)
        lj.index = i
        lj.tau2 = tree.ljet_tau2[i]
        lj.tau3 = tree.ljet_tau3[i]
        lj.tau32 = tree.ljet_tau32[i]
        #lj.tau21 = tree.ljet_tau21[i]
        #lj.sd12  = tree.ljet_sd12[i]
        #lj.sd23  = tree.ljet_sd23[i]
        #lj.Qw    = tree.ljet_Qw[i]
        #lj.ntrk  = tree.ljet_QGTaggerNTrack[i]

    return ljets


###############################


def RotateJets(ljets=[], phi=None):
    if phi == None:
        phi = -ljets[0].Phi()

    dPhi = ljets[0].DeltaPhi(ljets[1])
    ljets[0].SetPhi(0)
    ljets[1].SetPhi( abs(dPhi ) )

#    for lj in ljets:
#        lj.RotateZ(phi)

    return phi

#~~~~~~~~~~~~~~~~~~~~~~


def FlipEta(ljets=[]):
    for lj in ljets:
        lj.SetPtEtaPhiE(lj.Pt(), -lj.Eta(), lj.Phi(), lj.E())

####################

def save_training_history( history, filename = "GAN/training_history.root", verbose=True ):
   training_root = TFile.Open( filename, "RECREATE" )

   if verbose == True:
      print "INFO: saving training history..."

   h_d_lr = TGraphErrors()
   h_g_lr = TGraphErrors()
   h_d_loss = TGraphErrors()
   h_d_loss_r = TGraphErrors()
   h_d_loss_f = TGraphErrors()
   h_d_acc = TGraphErrors()
   h_g_loss = TGraphErrors()
   h_d_acc = TGraphErrors()
   h_d_acc_f = TGraphErrors()
   h_d_acc_r = TGraphErrors()

   n_epochs = len(history['d_loss'])
   for i in range(n_epochs):
       d_lr = history['d_lr'][i]
       g_lr = history['g_lr'][i]
       d_loss = history['d_loss'][i]
       d_loss_r = history['d_loss_r'][i]
       d_loss_f = history['d_loss_f'][i]
       d_acc = history['d_acc'][i]
       d_acc_f = history['d_acc_f'][i]
       d_acc_r = history['d_acc_r'][i]
       g_loss = history['g_loss'][i]

       h_d_lr.SetPoint(i, i, d_lr)
       h_g_lr.SetPoint(i, i, g_lr)
       h_d_loss.SetPoint(i, i, d_loss)
       h_d_loss_r.SetPoint(i, i, d_loss_r)
       h_d_loss_f.SetPoint(i, i, d_loss_f)
       h_d_acc.SetPoint(i, i, d_acc)
       h_d_acc_f.SetPoint(i, i, d_acc_f)
       h_d_acc_r.SetPoint(i, i, d_acc_r)
       h_g_loss.SetPoint(i, i, g_loss)

   h_d_lr.Write("d_lr")
   h_g_lr.Write("g_lr")
   h_d_loss.Write("d_loss")
   h_d_loss_r.Write("d_loss_r")
   h_d_loss_f.Write("d_loss_f")
   h_g_loss.Write("g_loss")
   h_d_acc.Write("d_acc")
   h_d_acc_f.Write("d_acc_f")
   h_d_acc_r.Write("d_acc_r")

   training_root.Write()
   training_root.Close()

   if verbose == True:
      print "INFO: training history saved to file:", training_root.GetName()
