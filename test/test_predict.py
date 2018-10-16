import keras.backend as K
from tensorflow.python import debug as tf_debug
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

from keras.models import load_model
mymodel = load_model("/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining/KERAS_model.h5")

from outputfeatures import features
print "Make prediction"
prediction = mymodel.predict( features )
print prediction
