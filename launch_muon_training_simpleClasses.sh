#!/bin/sh -x

#select training name
prefix='TTs_Muon2_simpleClasses_'

#select training data and 
trainingDataTxtFile='/local/gmoertl/DeepLepton/TrainingData/v6/step3/2016/muo/pt_5_-1/TTs/train_muo.txt'    #txt file should contain all training files, files should be stored in the same directroy as the txt file
trainingOutputDirectory='/local/gmoertl/DeepLepton/DeepJet_GPU/DeepJetResults'                              #training output directory must exist
trainingDataStructure='TrainData_deepLeptons_Muons_sorted_2016_simple'                                      #select from DeepJet/modules/datastructures/TrainData_deepLeptons.py
trainingModelReference='Train/deepLepton_Muons_reference.py'                                                #select from DeepJet/Train/deepLeptonXYreference.py, where the training model can be defined, define architecture in DeepJet/modules/models/convolutional_deepLepton.py and layers in DeepJet/modules/models/buildingBlocks_deepLepton.py

#select evaluation data
EvaluationTestDataTxtFile='/local/gmoertl/DeepLepton/TrainingData/v6/step3/2016/muo/pt_5_-1/TTs/50test_muo.txt' 
EvaluationTrainDataTxtFile='/local/gmoertl/DeepLepton/TrainingData/v6/step3/2016/muo/pt_5_-1/TTs/50train_muo.txt'  


#0) Source Environment:
#source ./gpu_env.sh
#ulimit -m unlimited; ulimit -v unlimited

#1) Preperation:
# Conversion to Data Structure
convertFromRoot.py -i ${trainingDataTxtFile} -o ${trainingOutputDirectory}/${prefix}TrainData -c ${trainingDataStructure}
# ensure, that all files have been converted
convertFromRoot.py -r ${trainingOutputDirectory}/${prefix}TrainData/snapshot.dc
convertFromRoot.py -r ${trainingOutputDirectory}/${prefix}TrainData/snapshot.dc

#2) Training:
python ${trainingModelReference} ${trainingOutputDirectory}/${prefix}TrainData/dataCollection.dc ${trainingOutputDirectory}/${prefix}Training

#3) Evaluation:
#a) for test data
convertFromRoot.py --testdatafor ${trainingOutputDirectory}/${prefix}Training/trainsamples.dc -i ${EvaluationTestDataTxtFile} -o ${trainingOutputDirectory}/${prefix}TestData
predict.py ${trainingOutputDirectory}/${prefix}Training/KERAS_model.h5 ${trainingOutputDirectory}/${prefix}TestData/dataCollection.dc ${trainingOutputDirectory}/${prefix}EvaluationTestData
#b) for train data
convertFromRoot.py --testdatafor ${trainingOutputDirectory}/${prefix}Training/trainsamples.dc -i ${EvaluationTrainDataTxtFile} -o ${trainingOutputDirectory}/${prefix}TestDataIsTrainData
predict.py ${trainingOutputDirectory}/${prefix}Training/KERAS_model.h5 ${trainingOutputDirectory}/${prefix}TestDataIsTrainData/dataCollection.dc ${trainingOutputDirectory}/${prefix}EvaluationTestDataIsTrainData

