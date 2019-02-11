echo Writing list for $1

mkdir -p filelists/$1/
ls ntuples_GAN/$1/tree.mg5_dijet_ht500.ptcl.pt250.nominal.root > filelists/$1/mg5_dijet_ht500.ptcl.pt250.GAN.txt
ls ntuples_GAN/$1/tree.mg5_dijet_ht500.reco.pt250.nominal.root > filelists/$1/mg5_dijet_ht500.reco.pt250.GAN.txt 

