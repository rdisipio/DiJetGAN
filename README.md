# DiJetGAN

## Check out package

```
git clone https://gitlab.cern.ch/disipio/DiJetGAN.git
cd DiJetGAN
```
## Create input files
The scripts accept as input a text file that contains the full path of ROOT files, which in turn are loaded to memory as a TChain object. 
It is assumed that ntuples are created using MadGraph5 + Pythia8 + Delphes3.

```
mkdir -p filelists
ls /home/disipio/mcgenerators/MG5_aMC_v2_6_4/pp2jj_lo/Events/run_04/tag_1_delphes_events.root > filelists/mg5_dijet_ht500.delphes.pt250.txt
```

The ROOT files create by Delphes3 are very large. Only a small fraction of the information is needed for the purpose of training the GAN. 
Thus, a smaller ntuples (using the "AnalysisTop mc16a" format) has to be created.

```
mkdir -p ntuples_MC
./delphes2tree.py -i filelists/mg5_dijet_ht500.delphes.pt250.txt -l reco
./delphes2tree.py -i filelists/mg5_dijet_ht500.delphes.pt250.txt -l ptcl
```

For your convenience, these files can be downloaded from CERNBOX: https://cernbox.cern.ch/index.php/s/AMA2Wa5jz96o6lW .
The files are stored in this folder: /eos/user/d/disipio/ntuples/dijet/


Now it is possible to convert ROOT file to CSV. This operation includes some pre-processing, e.g. all jets are phi-rotated by the same amount
so that the leading jet phi is always zero:

```
mkdir -p csv
ls ntuples_MC/tree.mg5_dijet_ht500.ptcl.pt250.nominal.root > filelists/mg5_dijet_ht500.ptcl.pt250.MC.txt
ls ntuples_MC/tree.mg5_dijet_ht500.reco.pt250.nominal.root > filelists/mg5_dijet_ht500.reco.pt250.MC.txt

./root2csv.py -i filelists/mg5_dijet_ht500.ptcl.pt250.MC.txt -l ptcl
./root2csv.py -i filelists/mg5_dijet_ht500.reco.pt250.MC.txt -l reco
```

## Initialize scaler
```
mkdir -p GAN
./init_scaler.py csv/mg5_dijet_ht500.reco.pt250.nominal.csv
./init_scaler.py csv/mg5_dijet_ht500.ptcl.pt250.nominal.csv 
```

## Train the generative-adversarial network (GAN): 

```
mkdir -p GAN
mkdir -p img
./train_GAN.py -e 5000 -d mg5_dijet_ht500 -l reco
./train_GAN.py -e 5000 -d mg5_dijet_ht500 -l ptcl
```
The generator model and the scaler have been saved to the GAN folder.

Plot training history:
```
./plot_traninig.py reco
./plot_traninig.py ptcl

# only if you have already created the MG5 histograms:
./plot_training_observables.py ptcl
./plot_training_observables.py reco
```

## Generate events

```
mkdir -p ntuples_GAN
./generate_events.py -l reco -n 500000
./generate_events.py -l ptcl -n 500000

ls ntuples_GAN/tree.mg5_dijet_ht500.ptcl.pt250.nominal.root > filelists/mg5_dijet_ht500.ptcl.pt250.GAN.txt
ls ntuples_GAN/tree.mg5_dijet_ht500.reco.pt250.nominal.root > filelists/mg5_dijet_ht500.reco.pt250.GAN.txt 
```

## Fill histograms

We want to compare three series of data:
* Original MG5 distributions, p4's are calculated using Lorentz vectors kinematics (tag: ```p4_tlv```)
* GAN-generated distributions, p4's are calculated using Lorentz vectors kinematics from the (pT,eta,phi,M) of the two leading jets (tag: ```p4_tlv```)
* GAN-generated distributions, all distributions are taken from the GAN's output (tag: ```p4_gan```)

```
mkdir -p histograms
./fill_histograms.py filelists/mg5_dijet_ht500.ptcl.pt250.MC.txt
./fill_histograms.py filelists/mg5_dijet_ht500.ptcl.pt250.GAN.txt

```

## Make final plots

```
cat observables.txt | parallel ./plot_observables.py {} ptcl mg5_dijet_ht500
cat observables.txt | parallel ./plot_observables.py {} reco mg5_dijet_ht500
```

You can do all the above with the following script:

```
./workflow.sh -d mg5_dijet_ht500 -l reco -e ${n_training_epochs} -n ${n_generate_events}
./workflow.sh -d mg5_dijet_ht500 -l ptcl -e ${n_training_epochs} -n ${n_generate_events}
```
