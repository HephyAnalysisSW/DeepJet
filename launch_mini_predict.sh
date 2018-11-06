#!/bin/sh -x

#trainingPath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181105/DYvsQCD_sorted_looseId_MuonTraining'
trainingPath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining'
testfilesTXT='/afs/hephy.at/data/gmoertl01/DeepLepton/predictions/check_flat_vs_full/TTJets/flat/tree.txt'
predictPath='/afs/hephy.at/data/gmoertl01/DeepLepton/predictions/check_flat_vs_full/TTJets/flat/'


#Convert
convertFromRoot.py --testdatafor ${trainingPath}/trainsamples.dc -i ${testfilesTXT} -o ${predictPath}/TestData
#Predict
predict.py ${trainingPath}/KERAS_model.h5 ${predictPath}/TestData/dataCollection.dc ${predictPath}/EvaluationTestData


