GeV = 1e3
TeV = 1e6

iLumi_2015 =  3219.56
iLumi_2016 = 32988.1
iLumi_2017 = 44307.4
iLumi_2018 = 36159.4
iLumi_All  = iLumi_2015 + iLumi_2016 + iLumi_2017 + iLumi_2018
iLumi      = iLumi_All

xs = {
  '410470' : 452.352426, # non all-had
  '410471' : 379.485909, # all-had
  '361023' : 8469.51264,
  '361024' : 135.3027431,
  '361025' : 4.207206325,
  '361026' : 0.242773531,
}

sumw_mc16a = {
 '410470' : 43808539236.0,
 '410471' : 14599590131.2,
  '361023' : 582209277.492,
  '361024' : 402892989.908,
  '361025' : 252804995.859,
  '361026' : 180202842.932,
}

sumw_mc16d = {
 '410470' : 54386909894.9,
 '410471' : 18045509444.8,
  '361023' : 1.,
  '361024' : 1.,
  '361025' : 1.,
  '361026' : 1.,
}

###########################

branches_eventInfo = [ 
 "eventNumber", "runNumber", "mcChannelNumber",
# "passed_*", "HLT_*",
 ]
branches_jets      = [
   "bjet_n_70", "jet_n",
   "jet_pt", "jet_eta", "jet_phi", "jet_e",
   "jet_mv2c10",
#   "jet_MV2c10rnn",
#   "jet_DL1",
#   "jet_DL1rnn",
   "jet_MV2cl100", # c vs l
   "jet_MV2c100",  # c vs b
   "jet_QGTaggerNTrack", # Number of associated tracks, pT > 50 GeV
 ]
branches_ljets    = [
  "ljet_pt", "ljet_eta", "ljet_phi", "ljet_e", "ljet_m",
  "ljet_tau2", "ljet_tau3", "ljet_tau32",
#  "ljet_tau21", "ljet_sd12", "ljet_sd23", "ljet_Qw",
#  "ljet_smoothedTopTaggerMassTau32_topTag80",
#  "ljet_QGTaggerNTrack",
 ]

branches_reco = [
  "reco_*",
]

branches_mc = [
  "weight_mc", "weight_pileup", "weight_jvt",
  "weight_bTagSF_MV2c10_70",
]

systematics_tree = [
   "nominal",
]
