#!/bin/bash

nepochs=5000
nevents=200000
dsid="mg5_dijet_ht500"
name="GAN_nominal_test"
level="reco" #ptcl
preselection="pt250"
while [ $# -gt 0 ] ; do
case $1 in
   -e) nepochs=$2       ; shift 2;;
   -n) nevents=$2       ; shift 2;;
   -d) dsid=$2          ; shift 2;;
   -o) name=$2          ; shift 2;;
   -l) level=$2         ; shift 2;;
esac
done

echo
echo "---------------"
echo "1) Training GAN"
echo "---------------"
echo

./train_GAN.py -e $nepochs -d $dsid -o $name -l $level
./plot_training.py -o $name -l $level -d $dsid

echo
echo "------------------"
echo "2) Generate events"
echo "------------------"
echo

./generate_events.py -d $dsid -n $nevents -l $level -o $name

echo
echo "------------------"
echo "3) Fill histograms"
echo "------------------"
echo

source makeGANFileLists.sh $name
./fill_histograms.py -f filelists/$name/$dsid.$level.$preselection.GAN.txt -o $name
./fill_histograms.py -f filelists/$dsid.$level.$preselection.MC.txt -o $name

echo
echo "-------------"
echo "3) Make plots"
echo "-------------"
echo

for obs in `cat observables.txt` 
do 
    echo "$obs"
    ./plot_observables.py -o $obs -l $level -n $name
done


echo
echo "-------------"
echo "4) Make training plots"
echo "-------------"
echo

for epoch in {0..$nepochs..250}
do 
    echo "$epoch"
    ./plot_training_observables.py -e $epoch -l $level -n $name
done
