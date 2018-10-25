import keras.backend as K
#from tensorflow.python import debug as tf_debug
#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

#files = [
#'/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5',
#'/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181014/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5',
#'/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181015/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5',
#'/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181016/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5',
#'/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181017/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5',
#'/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181021/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5',
#]

from keras.models import load_model
#mymodel = load_model("/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5")
# new training
#mymodel = load_model("/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/TestTraining/KERAS_model.h5")

#for f in files:
#    m = load_model(f)
#    print f
#    print m.get_weights()

from keras.models import model_from_json
import pickle
import numpy as np
mymodel = model_from_json( pickle.load(file('model.pkl'))) 
weights = pickle.load(file('weights.pkl'))
#_weights = [np.nan_to_num(weights[0])] + weights[1:]
_weights = map( np.nan_to_num, weights)
mymodel.set_weights(_weights)
#
from multi_l_features import features
##from new_training_features import features
print "Make prediction"
prediction = mymodel.predict( features )
print prediction
