import pickle
import ROOT
import numpy as np

dcFilePath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTrainData'
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
        if means[0][i] == 0. and means[1][i] == 1.:
            print means.dtype.names[i], means[0][i], means[1][i]
            meansDict.update({means.dtype.names[i]: [means[0][i], means[1][i]]})



#iPath='/afs/hephy.at/data/gmoertl01/DeepLepton/trainfiles/v1/2016/muo/pt_15_to_inf/DYVsQCD_ptRelSorted/mini_modulo_0_trainfile_1.root'
#iFile = ROOT.TFile.Open(iPath, 'read')
#iTree = iFile.Get('tree')
#nEntries = iTree.GetEntries()
#
#branchList = [
#'lep_pt',
#'lep_eta',
#'pfCand_neutral_ptRel_ptRelSorted',
#'pfCand_charged_ptRel_ptRelSorted',
#]
#
#
#for branch in branchList:
#    valList = []
#
#    for i in xrange(nEntries):
#        iTree.GetEntry(i)
#        val = iTree.GetLeaf(branch)
#        valList.append([])
#        for j in xrange(val.GetLen()):
#            valList[i].append((val.GetValue(j)-meansDict[branch][0])/meansDict[branch][1])
#
#    print '%s %.10f %.10f %.7e' %(branch, meansDict[branch][0], meansDict[branch][1], -meansDict[branch][0]/meansDict[branch][1])
#    for val in valList:
#        print val
#
#iFile.Close()

