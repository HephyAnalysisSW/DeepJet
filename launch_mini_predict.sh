#!/bin/sh -x

trainingPath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTraining'
#testfilesTXT='/afs/hephy.at/data/gmoertl01/DeepLepton/trainfiles/v1/2016/muo/pt_15_to_inf/DYVsQCD_ptRelSorted/mini_test_muo_std.txt'
testfilesTXT='/local/gmoertl/DeepLepton/TrainingData/v1/2016/muo/noPtSelection/TestSample_ptRelSorted/test_muo_std.txt'
predictPath='/local/gmoertl/DeepLepton/DeepJet_GPU/predict_byDeepJet_flatNtuple'

##Convert mini file
#convertFromRoot.py -i ${testfilesTXT} -o ${predictPath}/TrainData_mini -c TrainData_deepLeptons_Muons_ptRelSorted_2016

#Convert
convertFromRoot.py --testdatafor ${trainingPath}/trainsamples.dc -i ${testfilesTXT} -o ${predictPath}/TestData
Predict
predict.py ${trainingPath}/KERAS_model.h5 ${predictPath}/TestData/dataCollection.dc ${predictPath}/EvaluationTestData


