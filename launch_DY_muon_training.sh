#!/bin/sh

year='2016'
version='v2'
flavour='Muon'
short='muo'
run='1'
fromrun='1'
ptSelection='pt_15_to_inf'
sampleSelection='DYvsQCD_sorted'
sorted='_sorted'
sampleSize=''
prefix=''
DNN=''
model=''
ntestfiles='50'

#0) Source Environment:
#source ./gpu_env.sh

#1) Preperation:
convertFromRoot.py -i /local/gmoertl/DeepLepton/TrainingData/${version}/${year}/${short}/${ptSelection}/${sampleSelection}/${sampleSize}train_${short}_std.txt -o /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}TrainData -c TrainData_deepLeptons_${flavour}s${sorted}_${year}
convertFromRoot.py -r /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${fromrun}TrainData/snapshot.dc

#2) Training:
python Train/deepLepton${flavour}s${DNN}_reference.py /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${fromrun}TrainData/dataCollection.dc /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}Training

#3) Evaluation:
#a) for test data
convertFromRoot.py --testdatafor /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}Training/trainsamples.dc -i /local/gmoertl/DeepLepton/TrainingData/${version}/${year}/${short}/${ptSelection}/${sampleSelection}/${ntestfiles}test_${short}_std.txt -o /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}TestData
predict.py /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}Training/KERAS${model}_model.h5 /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}TestData/dataCollection.dc /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}EvaluationTestData
#b) for train data
convertFromRoot.py --testdatafor /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}Training/trainsamples.dc -i /local/gmoertl/DeepLepton/TrainingData/${version}/${year}/${short}/${ptSelection}/${sampleSelection}/${ntestfiles}train_${short}_std.txt -o /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}TestDataIsTrainData
predict.py /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}Training/KERAS_${model}model.h5 /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}TestDataIsTrainData/dataCollection.dc /local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults/${sampleSelection}_${prefix}${flavour}${run}EvaluationTestDataIsTrainData


#4) Plots:
#a) for test data
#cd /afs/cern.ch/work/g/gmortl/DeepJet/MuonEvaluationTestData
#python /afs/cern.ch/work/g/gmortl/DeepJet/DeepJet/Train/Plotting/ROC_lepton.py
#cd /afs/cern.ch/work/g/gmortl/DeepJet/DeepJet
#b) for train data
#cd /afs/cern.ch/work/g/gmortl/DeepJet/MuonEvaluationTrainData
#python /afs/cern.ch/work/g/gmortl/DeepJet/DeepJet/Train/Plotting/ROC_lepton.py
#cd /afs/cern.ch/work/g/gmortl/DeepJet/DeepJet
