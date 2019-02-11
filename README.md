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
./train_GAN.py -e 5000 -d mg5_dijet_ht500 -l reco -o GAN
./train_GAN.py -e 5000 -d mg5_dijet_ht500 -l ptcl -o GAN
```
The generator model is saved in the folder specified with the -o option while the scaler was saved in the "GAN" folder.

Plot training history:
```
./plot_traninig.py -l reco -d mg5_dijet_ht500 -o GAN
./plot_traninig.py -l ptcl -d mg5_dijet_ht500 -o GAN

# only if you have already created the MG5 histograms:
./plot_training_observables.py -l ptcl -e 500 -n GAN
./plot_training_observables.py -l reco -e 500 -n GAN
```

The -e specifies the epoch number that will be plotted. The -n option defined the GAN name used also for defining the output folder.

## Generate events

```
mkdir -p ntuples_GAN
./generate_events.py -l reco -n 500000 -o GAN_name
./generate_events.py -l ptcl -n 500000 -o GAN_name

The events will be stored in the folder ntuples_GAN/GAN_name 
Then create the file list using the script 

source makeGANFileLists.sh $name

Where $name is the GAN name used in the -o option

```

## Fill histograms

We want to compare three series of data:
* Original MG5 distributions, p4's are calculated using Lorentz vectors kinematics (tag: ```p4_tlv```)
* GAN-generated distributions, p4's are calculated using Lorentz vectors kinematics from the (pT,eta,phi,M) of the two leading jets (tag: ```p4_tlv```)
* GAN-generated distributions, all distributions are taken from the GAN's output (tag: ```p4_gan```)

```
mkdir -p histograms
./fill_histograms.py -f filelists/GAN_name/$dsid.$level.$preselection.GAN.txt -o GAN_name
./fill_histograms.py -f filelists/$dsid.$level.$preselection.MC.txt -o GAN_name

Note the different folder structure for GAN and MC. The output will be stored for both MC anf GAN in histograms/GAN_name folder.

```

## Make final plots

```
cat observables.txt | parallel ./plot_observables.py -o {} -l ptcl -d mg5_dijet_ht500 -n GAN_name
cat observables.txt | parallel ./plot_observables.py -o {} -l reco -d mg5_dijet_ht500 -n GAN_name
```

All plots will be stored in the img/GAN_name folder.

You can do all the above with the following script:

```
./workflow.sh -d mg5_dijet_ht500 -l reco -e ${n_training_epochs} -n ${n_generate_events} -o GAN_name
./workflow.sh -d mg5_dijet_ht500 -l ptcl -e ${n_training_epochs} -n ${n_generate_events} -o GAN_name
```

In this case, 