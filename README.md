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
ls /home/disipio/mcgenerators/MG5_aMC_v2_6_4/pp2jj_lo/Events/run_04/tag_1_delphes_events.root > filelists/delphes.mg5_dijet_ht500.MC.incl.txt
```

The ROOT files create by Delphes3 are very large. Only a small fraction of the information is needed for the purpose of training the GAN. 
Thus, a smaller ntuples (using the "AnalysisTop mc16a" format) has to be created.

```
./delphes2at.py -i filelists/delphes.mg5_dijet_ht500.MC.incl.txt 
```

For your convenience, this file can be downloaded from CERNBOX: https://cernbox.cern.ch/index.php/s/cXjogAvrojdUQ3f .


Now it is possible to convert ROOT file to CSV. This operation includes some pre-processing, e.g. all jets are phi-rotated by the same amount
so that the leading jet phi is always zero:

```
mkdir -p csv
./root2csv.py -i filelists/mc16a.mg5_dijet_ht500.MC.incl.txt
```

## Create scaler
```
./create_scaler.py csv/mc16a.mg5_dijet_ht500.rnn.GAN.incl.nominal.csv
```

## Train the generative-adversarial network (GAN): 

```
mkdir -p GAN
mkdir -p img
./train_GAN.py -e 5000 -d mg5_dijet_ht500
```
The generator model and the scaler have been saved to the GAN folder.

Plot training history:
```
./plot_traninig.py
```

## Generate events

```
./generate_events.py -n 500000
ls GAN/tree.mg5_dijet_ht500.rnn.GAN.incl.nominal.root > filelists/mc16a.mg5_dijet_ht500.GAN.incl.txt
```

## Fill histograms

```
mkdir -p histograms
./fill_histograms.py filelists/mc16a.mg5_dijet_ht500.GAN.incl.txt 
./fill_histograms.py filelists/mc16a.mg5_dijet_ht500.MC.incl.txt 
```

## Make final plots

```
cat observables.txt | parallel ./plot_observables.py {} mg5_dijet_ht500
```

You can do all the above with the following script:

```
./workflow.sh -d mg5_dijet_ht500 -e ${n_training_epochs} -n ${n_generate_events}
```
