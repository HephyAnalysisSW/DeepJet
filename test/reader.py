#Standard imports

import pickle
import ROOT
import numpy as np
import os
import shutil
import uuid
import operator
        
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

        self._feature_getters = {}

        print "Branches:"
        for i in range(len(self.branches)):
            print "Collection", i
            for i_b, b in enumerate(self.branches[i]):
                print "  branch %2i/%2i %40s   mean %8.5f var %8.5f" %( i, i_b, b, self.means_dict[b][0], self.means_dict[b][1])
            print 

class InputData:

    def __init__( self, filename, treename = "tree"):

        # read class from file
        file_= ROOT.TFile( filename )
        tree = file_.Get(treename)
        # tmp locations
        self.tmpdir = "."
        self.tmpname = "DL_reader_"+uuid.uuid4().hex
        self.tmp_filenames = [ "%s.C"%self.tmpname, "%s.h"%self.tmpname ]
        # make class
        tree.MakeClass( self.tmpname )
        file_.Close()

        # move files to tmp area
        for file in self.tmp_filenames:
            shutil.move( file, os.path.join( self.tmpdir, file ) )

        # load the newly created files as macro
        ROOT.gROOT.LoadMacro( os.path.join( self.tmpdir, self.tmpname+'.C' ) )

        # make chain (can be used with more files)
        self.chain = ROOT.TChain( treename )
        self.chain.Add( filename )

        # make instance
        self.event = getattr(ROOT, "%s" % self.tmpname )( self.chain )

        self.chain.GetEntries()
        self.init_getters()
 
    # Clean up the tmp files
    def __del__( self ):
        import os #Interesting. os gets un-imported in the destructor :-)
        for file_ in self.tmp_filenames:
           filename = os.path.join( self.tmpdir, file_ )
           if os.path.exists( filename ):
                os.remove( filename )

    def getEntry( self, nevent ):
        self.chain.GetEntry( nevent )

    # Store a list of functors that retrieve the correct branch from the event
    def init_getters( self):
        self._getters = {'neutrals': {
            "pfCand_neutral_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_pt"),
        "pfCand_neutral_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_dxy_pf"),
         "pfCand_neutral_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_dz_pf"),
   "pfCand_neutral_puppiWeight_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_puppiWeight"),
  "pfCand_neutral_hcalFraction_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_hcalFraction"),
        "pfCand_neutral_fromPV_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_fromPV"),
        },
                         'charged': {
            "pfCand_charged_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_pt"),
        "pfCand_charged_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_dxy_pf"),
         "pfCand_charged_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_dz_pf"),
   "pfCand_charged_puppiWeight_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_puppiWeight"),
  "pfCand_charged_hcalFraction_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_hcalFraction"),
        "pfCand_charged_fromPV_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_fromPV"),
        },
                         'photon': {
            "pfCand_photon_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_pt"),
        "pfCand_photon_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_dxy_pf"),
         "pfCand_photon_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_dz_pf"),
   "pfCand_photon_puppiWeight_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_puppiWeight"),
  "pfCand_photon_hcalFraction_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_hcalFraction"),
        "pfCand_photon_fromPV_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_fromPV"),
        },
                         'electron': {
          "pfCand_electron_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_pt"),
      "pfCand_electron_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_dxy_pf"),
       "pfCand_electron_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_dz_pf"),
        },
                         'muon': {
              "pfCand_muon_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_pt"),
          "pfCand_muon_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_dxy_pf"),
           "pfCand_muon_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_dz_pf"),
        },
    }

    # Store a list of functors that retrieve the correct branch from the event
    def feature_getters( self, collection_name):
        # return getters, if collection_name known, otherwise create      
        if self._feature_getters.has_key(collection_name): return self._feature_getters[collection_name] 
        self._feature_getters[collection_name] = {
                          "lep_pt":operator.attrgetter(collection_name+'_pt'),
                         "lep_eta":operator.attrgetter(collection_name+'_eta'),
                         "lep_phi":operator.attrgetter(collection_name+'_phi'),
                         "lep_dxy":operator.attrgetter(collection_name+'_dxy'),
                          "lep_dz":operator.attrgetter(collection_name+'_dz'),
                        "lep_edxy":operator.attrgetter(collection_name+'_edxy'),
                         "lep_edz":operator.attrgetter(collection_name+'_edz'),
                        "lep_ip3d":operator.attrgetter(collection_name+'_ip3d'),
                       "lep_sip3d":operator.attrgetter(collection_name+'_sip3d'),
              "lep_innerTrackChi2":operator.attrgetter(collection_name+'_innerTrackChi2'),
  "lep_innerTrackValidHitFraction":operator.attrgetter(collection_name+'_innerTrackValidHitFraction'),
                     "lep_ptErrTk":operator.attrgetter(collection_name+'_ptErrTk'),
                         "lep_rho":operator.attrgetter(collection_name+'_rho'),
                       "lep_jetDR":operator.attrgetter(collection_name+'_jetDR'),
         "lep_trackerLayers_float":operator.attrgetter(collection_name+'_trackerLayers'),
           "lep_pixelLayers_float":operator.attrgetter(collection_name+'_pixelLayers'),
           "lep_trackerHits_float":operator.attrgetter(collection_name+'_trackerHits'),
              "lep_lostHits_float":operator.attrgetter(collection_name+'_lostHits'),
         "lep_lostOuterHits_float":operator.attrgetter(collection_name+'_lostOuterHits'),
                    "lep_relIso03":operator.attrgetter(collection_name+'_relIso03'),
           "lep_miniRelIsoCharged":operator.attrgetter(collection_name+'_miniRelIsoCharged'),
           "lep_miniRelIsoNeutral":operator.attrgetter(collection_name+'_miniRelIsoNeutral'),
                "lep_jetPtRatiov1":operator.attrgetter(collection_name+'_jetPtRatiov1'),
                  "lep_jetPtRelv1":operator.attrgetter(collection_name+'_jetPtRelv1'),
        "lep_segmentCompatibility":operator.attrgetter(collection_name+'_segmentCompatibility'),
          "lep_muonInnerTrkRelErr":operator.attrgetter(collection_name+'_muonInnerTrkRelErr'),
          "lep_isGlobalMuon_float":operator.attrgetter(collection_name+'_isGlobalMuon'),
           "lep_chi2LocalPosition":operator.attrgetter(collection_name+'_chi2LocalPosition'),
           "lep_chi2LocalMomentum":operator.attrgetter(collection_name+'_chi2LocalMomentum'),
             "lep_globalTrackChi2":operator.attrgetter(collection_name+'_globalTrackChi2'),
         "lep_glbTrackProbability":operator.attrgetter(collection_name+'_glbTrackProbability'),
                     "lep_trkKink":operator.attrgetter(collection_name+'_trkKink'),
           "lep_caloCompatibility":operator.attrgetter(collection_name+'_caloCompatibility'),
             "lep_nStations_float":operator.attrgetter(collection_name+'_nStations'),
        }
        return self._feature_getters[collection_name] 

    # for a given lepton, read the pf candidate mask and return the PF indices
    def get_pf_indices( self, pf_type, collection_name, n_lep):
        n = getattr(self.event, "nDL_pfCand_%s"%pf_type )
        pf_mask = getattr(self.event, "DL_pfCand_%s_%s_mask" % (pf_type, "selectedLeptons" if collection_name=="LepGood" else "otherLeptons" ) )
        mask_ = (1<<n_lep)
        return filter( lambda i: mask_&pf_mask[i], range(n))

    # put all inputs together
    def read_inputs( self, collection_name, branches, n_lep):
        
        # read the lepton features
        features = [ self.feature_getters( collection_name )[b](self.event)[n_lep] for b in branches ]

        # read pf indices
        pf_neutral_indices  = self.get_pf_indices( 'neutral', collection_name, n_lep )
        pf_charged_indices  = self.get_pf_indices( 'charged', collection_name, n_lep )
        pf_photon_indices   = self.get_pf_indices( 'photon', collection_name, n_lep )
        pf_muon_indices     = self.get_pf_indices( 'muon', collection_name, n_lep )
        pf_electron_indices = self.get_pf_indices( 'electron', collection_name, n_lep )



#         pfCand_neutral_ptRel_ptRelSorted
#        pfCand_neutral_deltaR_ptRelSorted
#            pfCand_neutral_pt_ptRelSorted
#        pfCand_neutral_dxy_pf_ptRelSorted
#         pfCand_neutral_dz_pf_ptRelSorted
#   pfCand_neutral_puppiWeight_ptRelSorted
#  pfCand_neutral_hcalFraction_ptRelSorted
#        pfCand_neutral_fromPV_ptRelSorted

#DL_pfCand_neutral_pt    pt for neutral pf candidates associated : 0 at: 0x478b850
#DL_pfCand_neutral_dxy_pf    dxy for neutral pf candidates associated : 0 at: 0x478e5a0
#DL_pfCand_neutral_dz_pf dz for neutral pf candidates associated : 0 at: 0x478ec10
#DL_pfCand_neutral_puppiWeight   puppiWeight for neutral pf candidates associated : 0 at: 0x478d210
#DL_pfCand_neutral_hcalFraction  hcalFraction for neutral pf candidates associated : 0 at: 0x478d8a0
#DL_pfCand_neutral_fromPV    fromPV for neutral pf candidates associated : 0 at: 0x478df30


#         pfCand_charged_ptRel_ptRelSorted
#        pfCand_charged_deltaR_ptRelSorted
#            pfCand_charged_pt_ptRelSorted
#        pfCand_charged_dxy_pf_ptRelSorted
#         pfCand_charged_dz_pf_ptRelSorted
#   pfCand_charged_puppiWeight_ptRelSorted
#  pfCand_charged_hcalFraction_ptRelSorted
#        pfCand_charged_fromPV_ptRelSorted
#
#          pfCand_photon_ptRel_ptRelSorted
#         pfCand_photon_deltaR_ptRelSorted
#             pfCand_photon_pt_ptRelSorted
#         pfCand_photon_dxy_pf_ptRelSorted
#          pfCand_photon_dz_pf_ptRelSorted
#    pfCand_photon_puppiWeight_ptRelSorted
#   pfCand_photon_hcalFraction_ptRelSorted
#         pfCand_photon_fromPV_ptRelSorted
#
#        pfCand_electron_ptRel_ptRelSorted
#       pfCand_electron_deltaR_ptRelSorted
#           pfCand_electron_pt_ptRelSorted
#       pfCand_electron_dxy_pf_ptRelSorted
#        pfCand_electron_dz_pf_ptRelSorted
#
#            pfCand_muon_ptRel_ptRelSorted
#           pfCand_muon_deltaR_ptRelSorted
#               pfCand_muon_pt_ptRelSorted
#           pfCand_muon_dxy_pf_ptRelSorted
#            pfCand_muon_dz_pf_ptRelSorted
#
#                                    SV_pt
#                                  SV_chi2
#                                  SV_ndof
#                                   SV_dxy
#                                  SV_edxy
#                                  SV_ip3d
#                                 SV_eip3d
#                                 SV_sip3d
#                              SV_cosTheta
#                                SV_deltaR
#                                 SV_jetPt
#                                SV_jetEta
#                                 SV_jetDR
#                          SV_maxDxyTracks
#                          SV_secDxyTracks
#                          SV_maxD3dTracks
#                          SV_secD3dTracks

if __name__ == "__main__": 
    # Information on the training
    training_directory = '/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTrainData'
    trainingInfo = TrainingInfo( training_directory )

    # Input data
    input_filename = "/afs/hephy.at/work/r/rschoefbeck/CMS/tmp/CMSSW_9_4_6_patch1/src/CMGTools/StopsDilepton/cfg/test/WZTo3LNu_amcatnlo_1/treeProducerSusySingleLepton/tree.root"
    inputData = InputData( input_filename )

    inputData.getEntry(0)
    res  = inputData.read_inputs("LepGood", trainingInfo.branches[0], 0)
    res2 = inputData.read_inputs("LepOther", trainingInfo.branches[0], 0)

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
