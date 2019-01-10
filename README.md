# QCDGAN

Check out package
```
git clone https://github.com/rdisipio/QCDGAN.git
```

Assuming ntuples are created with AnalysisTop + TTbarDiffXsTools

Create input file list

```
mkdir -p filelists
ls ntuples/user.pjacka.361024.Pythia8EvtGen.DAOD_TOPQ1.e3668_s3126_r9364_r9315_p3404.AT-21.2.29-MC16aDijet-v6_allhad_boosted_root/* > filelists/mc16a.361024.MC.incl.txt
```

Convert ROOT file to CSV. This operation includes some pre-processing:

```
./root2csv.py -i filelists/mc16a.361024.incl.txt 
```

Train the generative-adversarial network (GAN): 

```
mkdir -p GAN
./train_GAN.py -e 5000 -d 361024
```
This saved the generator model and the scaler.

Plot training history:
```
mkdir -p img
./plot_traninig.py
```

Generate events:
```
./generate_events.py -n 200000
ls GAN/tree.361024.rnn.GAN.incl.nominal.root > filelists/mc16a.361024.GAN.incl.txt
```

Fill histograms:
```
./tree2hist.py  filelists/mc16a.361024.GAN.incl.txt 200000
./tree2hist.py  filelists/mc16a.361024.MC.incl.txt 200000
```

Make final plots:
```
cat observables.txt | parallel ./plot_observables.py {} 361024
```
