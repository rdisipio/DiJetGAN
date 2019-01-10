#!/bin/bash

nepochs=3000
nevents=200000
dsid=361024
while [ $# -gt 0 ] ; do
case $1 in
   -e) nepochs=$2       ; shift 2;;
   -n) nevents=$2       ; shift 2;;
   -d) dsid=$2          ; shift 2;;
esac
done

./GAN.py -e $nepochs -d $dsid

./event_generator.py -n $nevents
./tree2hist.py  filelists/mc16a.$dsid.GAN.incl.txt $nevents
cat observables.txt | parallel ./plot_observables.py {} $dsid
