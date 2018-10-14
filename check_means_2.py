import pickle
import ROOT
import numpy as np

dcFilePath='/local/gmoertl/DeepLepton/DeepJet_GPU/predictTestData/TrainData_mini'
fd=open(dcFilePath+'/dataCollection.dc', 'rb')

samples=pickle.load(fd)
sampleentries=pickle.load(fd)
originRoots=pickle.load(fd)
nsamples=pickle.load(fd)
useweights=pickle.load(fd)
batchsize=pickle.load(fd)
dataclass=pickle.load(fd)
weighter=pickle.load(fd)
means=pickle.load(fd)
fd.close()

meansDict={}

for i in xrange(len(means[0])):
        meansDict.update({means.dtype.names[i]: [means[0][i], means[1][i]]})

iPath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainfiles/v1/2016/muo/pt_15_to_inf/DYVsQCD_ptRelSorted/mini_modulo_0_trainfile_1.root'

iFile = ROOT.TFile.Open(iPath, 'read')
iTree = iFile.Get('tree')
nEntries = iTree.GetEntries()

for branch in meansDict:
    valList = []

    for i in xrange(nEntries):
        iTree.GetEntry(i)
        val = iTree.GetLeaf(branch)
        for j in xrange(val.GetLen()):
            valList.append(val.GetValue(j))
    std = 1. if np.std(valList)==0 else np.std(valList)
    meansList = [np.mean(valList), std]
    diffList  = [meansDict[branch][0]-np.mean(valList), meansDict[branch][1]-std]
    print 'branchname: ', branch
    print 'values from pickle: \t', meansDict[branch]
    print 'values from root: \t', meansList
    print 'difference: \t\t', diffList
    print '\n'
