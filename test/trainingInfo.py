#Standard imports

import pickle
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
        self._means     =   pickle.load(file_)
        file_.close()


        # Get means dictionary
        self.means = {name : (self._means[0][i], self._means[1][i]) for i, name in enumerate( self._means.dtype.names) }

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
                print "  branch %2i/%2i %40s   mean %8.5f var %8.5f" %( i, i_b, b, self.means[b][0], self.means[b][1])
            print

    def dump( self, filename):
        pickle.dump( [ self.branches, self.means], file( filename, 'w' ) )
        print "Written", filename

if __name__ == "__main__": 
    # Information on the training
    #training_directory = '/afs/hephy.at/data/rschoefbeck01/DeepLepton/trainings/DYVsQCD_ptRelSorted_MuonTrainData/'
    training_directory = '/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181204/TTs_balanced_pt5toInf_MuonTrainData'
    trainingInfo = TrainingInfo( training_directory )
    trainingInfo.dump( os.path.join( training_directory, 'branches_means_vars.pkl' ) )
