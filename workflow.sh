#!/bin/bash

nepochs=5000
nevents=200000
dsid=361024
while [ $# -gt 0 ] ; do
case $1 in
   -e) nepochs=$2       ; shift 2;;
   -n) nevents=$2       ; shift 2;;
   -d) dsid=$2          ; shift 2;;
esac
done

echo
echo "---------------"
echo "1) Training GAN"
echo "---------------"
echo

./train_GAN.py -e $nepochs -d $dsid
./plot_training.py $dsid

echo
echo "------------------"
echo "2) Generate events"
echo "------------------"
echo

./generate_events.py -d $dsid -n $nevents

echo
echo "------------------"
echo "3) Fill histograms"
echo "------------------"
echo

./fill_histograms.py  filelists/mc16a.$dsid.GAN.incl.txt $nevents

echo
echo "-------------"
echo "3) Make plots"
echo "-------------"
echo

cat observables.txt | parallel ./plot_observables.py {} $dsid
