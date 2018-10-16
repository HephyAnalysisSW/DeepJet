#Standard imports

import pickle
import ROOT
import numpy as np
import os
        
# DeepJet & DeepJetCore
from DeepJetCore.DataCollection import DataCollection

class TrainingInfo:

    def __init__( self, directory ):

        filename = os.path.join( directory, 'dataCollection.dc')
        file_    = open( filename, 'rb')

        self.samples    =   pickle.load(file_)
        sampleentries   =   pickle.load(file_)
        originRoots     =   pickle.load(file_)
        nsamples        =   pickle.load(file_)
        useweights      =   pickle.load(file_)
        batchsize       =   pickle.load(file_)
        dataclass       =   pickle.load(file_)
        weighter        =   pickle.load(file_)
        self.means           =   pickle.load(file_)
        file_.close()


        # Get means dictionary
        self.means_dict = {name : (self.means[0][i], self.means[1][i]) for i, name in enumerate( self.means.dtype.names) }

        # Get DeepJetCore DataCollection
        self.dataCollection = DataCollection()
        self.dataCollection.readFromFile(filename) 

        # Reading first sample & get branch structure
        fullpath = self.dataCollection.getSamplePath(self.samples[0])
        self.dataCollection.dataclass.readIn(fullpath)
        self.branches = self.dataCollection.dataclass.branches

        print "Branches:"
        for i in range(len(self.branches)):
            print "Collection", i
            for i_b, b in enumerate(self.branches[i]):
                print "  branch %2i/%2i %40s   mean %8.5f var %8.5f" %( i, i_b, b, self.means_dict[b][0], self.means_dict[b][1])
            print 


if __name__ == "__main__": 
    # Information on the training
    training_directory = '/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTrainData'
    trainingInfo = TrainingInfo( training_directory )
    #


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
#
