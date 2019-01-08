header_GAN = [
  "runNumber", "eventNumber", "weight",
  "ljet1_px", "ljet1_py", "ljet1_pz", "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M", "ljet1_tau2", "ljet1_tau3", "ljet1_tau32",
  "ljet2_px", "ljet2_py", "ljet2_pz", "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M", "ljet2_tau2", "ljet2_tau3", "ljet2_tau32",
  "category"
]

features_GAN = [
  "ljet1_px", "ljet1_py", "ljet1_pz", "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M", "ljet1_tau2", "ljet1_tau3", "ljet1_tau32",
  "ljet2_px", "ljet2_py", "ljet2_pz", "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M", "ljet2_tau2", "ljet2_tau3", "ljet2_tau32",
]

n_fso_max = 2
n_features = len(features_GAN)
n_features_per_fso = int( n_features / n_fso_max )

