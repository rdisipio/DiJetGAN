from common import GeV, TeV
from ROOT import *
from array import array

import cPickle as pickle
from keras.models import load_model

known_classifiers = [ "rnn:GAN", "rnn:highlevel", "rnn:PxPyPzMBwNtrk" ]

parser = argparse.ArgumentParser(description="GAN event generator")
parser.add_argument( '-c', '--classifier',        default=known_classifiers[0] )
parser.add_argument( '-p', '--preselection',      default="incl" )
parser.add_argument( '-s', '--systematic',        default="nominal" ) 
parser.add_argument( '-d', '--dsid',              default="361024" )
parser.add_argument( '-n', '--nevents',           default=10000 )
args         = parser.parse_args()

classifier        = args.classifier
training_filename = args.training_filename
preselection      = args.preselection
systematic        = args.systematic
dsid              = args.dsid
n_events          = int(args.nevents)

ljets_n_max = 2

##################
# Load Keras stuff
scaler = None
dnn    = None
print "INFO: Systematic:", syst
print "INFO: Using classifier:", ( classifier_arch, classifier_feat )

model_filename  = "GAN/generator.%s.%s.%s.%s.%s.h5" % (dsid,classifier_arch, classifier_feat, preselection, systematic)
scaler_filename = "GAN/scaler.%s.%s.%s.%s.%s.pkl" % (dsid,classifier_arch, classifier_feat, preselection, systematic)

print "INFO: loading model from", model_filename
dnn = load_model( model_filename )
print dnn.summary()

print "INFO: loading scaler from", scaler_filename
with open( scaler_filename, "rb" ) as file_scaler:
  scaler    = pickle.load( file_scaler )

outfname = "GAN/histograms.%s.%s.%s.%s.%s.root" % (dsid,classifier_arch, classifier_feat, preselection, systematic)  
outfile = TFile.Open( outfname, "RECREATE" )

b_eventNumber     = array( 'l', [ 0 ] )
b_abcd16          = array( 'i', [0] )
b_ljet_px = vector('float')(2)
b_ljet_py = vector('float')(2)
b_ljet_pz = vector('float')(2)
b_ljet_pt = vector('float')(2)
b_ljet_eta = vector('float')(2)
b_ljet_phi = vector('float')(2)
b_ljet_E   = vector('float')(2)
b_ljet_m  = vector('float')(2)
b_ljet_tau2  = vector('float')(2)
b_ljet_tau3  = vector('float')(2)
b_ljet_tau32 = vector('float')(2)
b_ljet_bmatch70_dR = vector('float')(2)
b_ljet_bmatch70    = vector('int')(2)
b_ljet_isTopTagged80  = vector('int')(2)

outtree = TTree( systematic, "GAN generated events" )
outtree.Branch( 'eventNumber',        b_eventNumber,     'eventNumber/l' )
outtree.Branch( 'ljet_px',            b_ljet_px )
outtree.Branch( 'ljet_py',            b_ljet_py )
outtree.Branch( 'ljet_pz',            b_ljet_pz )
outtree.Branch( 'ljet_pt',            b_ljet_pt )
outtree.Branch( 'ljet_eta',           b_ljet_eta )
outtree.Branch( 'ljet_phi',           b_ljet_phi )
outtree.Branch( 'ljet_E',             b_ljet_E  )
outtree.Branch( 'ljet_m',             b_ljet_m  )
outtree.Branch( 'ljet_tau2',          b_ljet_tau2 )
outtree.Branch( 'ljet_tau3',          b_ljet_tau2 )
outtree.Branch( 'ljet_tau32',         b_ljet_tau32 )
outtree.Branch( 'ljet_bmatch70_dR',   b_ljet_bmatch70_dR )
outtree.Branch( 'ljet_bmatch70',      b_ljet_bmatch70 )
#outtree.Branch( 'ljet_isTopTagged80', b_ljet_isTopTagged80 )
outtree.Branch( 'ljet_smoothedTopTaggerMassTau32_topTag80',  b_ljet_isTopTagged80 )
outtree.Branch( 'abcd16',             b_abcd16, 'abcd16/I' )

print "INFO: generating %i events..." % n_events

X_noise = np.random.uniform(0,1,size=[ n_events, GAN_input_size])
X_generated = generator.predict(X_noise)

print "INFO: generated %i events" % n_events

X_generated = scaler.inverse_transform( X_generated )
   
print "INFO: ...done."
print

print "INFO: starting event loop:", n_events

for ievent in range(n_events):
   if ( n_events < 10 ) or ( (ievent+1) % int(float(n_events)/10.)  == 0 ):
      perc = 100. * ievent / float(n_events)
      print "INFO: Event %-9i  (%3.0f %%)" % ( ievent, perc )

   # event weight
   w = 1.0

   b_eventNumber[0] = ievent

   ljets = []
   for i in range(ljets_n_max):
      ljets += [ TLorentzVector() ]
      lj = ljets[-1]
   
      pt    = X_generated[ievent][i][0]
      eta   = X_generated[ievent][i][1]
      phi   = X_generated[ievent][i][2]
      E     = max( 0., X_generated[ievent][i][3] )
      m     = max( 0., X_generated[ievent][i][4] )
      #tau32 = -1. #max( X_generated[ievent][i][4], 0. )
      lj.SetPtEtaPhiM( pt, eta, phi, m )
           
      #px    = X_generated[ievent][i][0]
      #py    = X_generated[ievent][i][1]
      #pz    = X_generated[ievent][i][2]
      #pt    = max( 0., X_generated[ievent][i][3] )
      #E     = max( 0., X_generated[ievent][i][3] )
      #m     = max( 0., X_generated[ievent][i][3] )
      #E     = TMath.Sqrt( px*px + py*py + pz*pz + m*m )
      #lj.SetPxPyPzE( px, py, pz, E )
     
      lj.tau2 = -1.
      lj.tau3 = -1.
      lj.tau32 = -1
      lj.isTopTagged80 = 0 # TopSubstructureTagger( lj )
      lj.bmatch70_dR = 10.
      lj.bmatch70    = -1

   # sort jets by pT
   ljets.sort( key=lambda jet: jet.Pt(), reverse=True )
   lj1 = ljets[0]
   lj2 = ljets[1]

    # Fill branches
   b_ljet_px[0]  = lj1.Px()*GeV
   b_ljet_py[0]  = lj1.Py()*GeV
   b_ljet_pz[0]  = lj1.Pz()*GeV
   b_ljet_pt[0]  = lj1.Pt()*GeV
   b_ljet_eta[0] = lj1.Eta()
   b_ljet_phi[0] = lj1.Phi()
   b_ljet_E[0]   = lj1.E()*GeV
   b_ljet_m[0]   = lj1.M()*GeV
   b_ljet_tau2[0]  = lj1.tau2
   b_ljet_tau3[0]  = lj1.tau3
   b_ljet_tau32[0] = lj1.tau32
   b_ljet_bmatch70_dR[0]   = lj1.bmatch70_dR
   b_ljet_bmatch70[0]      = lj1.bmatch70
   b_ljet_isTopTagged80[0] = lj1.isTopTagged80

   b_ljet_px[1]  = lj2.Px()*GeV
   b_ljet_py[1]  = lj2.Py()*GeV
   b_ljet_pz[1]  = lj2.Pz()*GeV
   b_ljet_pt[1]  = lj2.Pt()*GeV
   b_ljet_eta[1] = lj2.Eta()
   b_ljet_phi[1] = lj2.Phi()
   b_ljet_E[1]   = lj2.E()*GeV
   b_ljet_m[1]   = lj2.M()*GeV
   b_ljet_tau2[1]  = lj2.tau2
   b_ljet_tau3[1]  = lj2.tau3
   b_ljet_tau32[1] = lj2.tau32
   b_ljet_bmatch70_dR[1]   = lj2.bmatch70_dR
   b_ljet_bmatch70[1]      = lj2.bmatch70
   b_ljet_isTopTagged80[1] = lj2.isTopTagged80
   
   b_abcd16[0] = 0
   if lj1.isTopTagged80 == True: b_abcd16[0] ^= 0b1000
   if lj1.bmatch70 > 0:          b_abcd16[0] ^= 0b0100
   if lj2.isTopTagged80 == True: b_abcd16[0] ^= 0b0010
   if lj2.bmatch70 > 0:          b_abcd16[0] ^= 0b0001

   outtree.Fill()

   # end event loop
   
outtree.Write()
outfile.Close()

print "INFO: output file created:", outfname
print "INFO: done."
