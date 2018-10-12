#!/bin/sh

year='2016'
flavour='Muon'
short='muo'
run='_test1_'
fromrun=''
ptSelection='pt_15_to_inf'
sampleSelection='DYVsQCD'
sampleSize='mini_'
prefix=''

trainingPath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining'
testfilesTXT='/afs/hephy.at/data/gmoertl01/DeepLepton/trainfiles/v1/2016/muo/pt_15_to_inf/DYVsQCD_ptRelSorted/mini_test_muo_std.txt'
predictPath='/local/gmoertl/DeepLepton/DeepJet_GPU/predictTestData'


#Convert
convertFromRoot.py --testdatafor ${trainingPath}/trainsamples.dc -i ${testfilesTXT} -o ${predictPath}/TestData
#Predict
mini_predict.py ${trainingPath}/KERAS_model.h5 ${predictPath}/TestData/dataCollection.dc ${predictPath}EvaluationTestData


