from keras.models import load_model
mymodel = load_model("/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5")

from outputfeatures import features
print "Make prediction"
prediction = mymodel.predict( features )
print prediction
