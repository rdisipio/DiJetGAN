#!/usr/bin/env python

import os, sys

from keras.utils import plot_model
from keras.models import load_model

dsid = "mg5_dijet_ht500"
classifier = "rnn:GAN"
classifier_arch, classifier_feat = classifier.split(':')
preselection = "incl"
systematic = "nominal"

if len(sys.argv) > 1: dsid = sys.argv[1]

model_filename  = "GAN/generator.%s.%s.%s.%s.%s.h5" % (dsid,classifier_arch, classifier_feat, preselection, systematic)
print "INFO: loading generator model from", model_filename

generator = load_model( model_filename )
print generator.summary()

img_filename = "img/model_%s.png" % ( dsid )

plot_model( generator, to_file=img_filename )
