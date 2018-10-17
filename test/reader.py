#Standard imports

import pickle
import ROOT
import numpy as np
import os
import shutil
import uuid
import operator
from math import *
        
# DeepJet & DeepJetCore
from DeepJetCore.DataCollection import DataCollection

def ptRel(p4,axis):
    a = ROOT.TVector3(axis.Vect().X(),axis.Vect().Y(),axis.Vect().Z())
    o = ROOT.TLorentzVector(p4.Px(),p4.Py(),p4.Pz(),p4.E())
    return o.Perp(a)

def deltaPhi(phi1, phi2):
    dphi = phi2-phi1
    if  dphi > pi:
        dphi -= 2.0*pi
    if dphi <= -pi:
        dphi += 2.0*pi
    return abs(dphi)

def deltaR2(eta1, phi1, eta2, phi2):
    return deltaPhi(phi1, phi2)**2 + (eta1 - eta2)**2

def deltaR(*args, **kwargs):
    return sqrt(deltaR2(*args, **kwargs))

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

class InputData:

    flavors = ['charged', 'neutral', 'photon', 'muon', 'electron']

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

        # dict cache for all pf candidates in the event
        self._pf_candidates = {} 
        self._nevent        = None

    # Clean up the tmp files
    def __del__( self ):
        import os #Interesting. os gets un-imported in the destructor :-)
        for file_ in self.tmp_filenames:
           filename = os.path.join( self.tmpdir, file_ )
           if os.path.exists( filename ):
                os.remove( filename )

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

    # Store a list of functors that retrieve the correct branch from the event
    def init_getters( self):
        self._feature_getters = {}
        self.pf_size_getters = { 
            "neutral":operator.attrgetter( "nDL_pfCand_neutral"),
            "charged":operator.attrgetter( "nDL_pfCand_charged"),
            "photon":operator.attrgetter( "nDL_pfCand_photon"),
            "muon":operator.attrgetter( "nDL_pfCand_muon"),
            "electron":operator.attrgetter( "nDL_pfCand_electron"),
            }
        self.pf_getters = { 'neutral':{
            "pfCand_neutral_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_pt"),
           "pfCand_neutral_eta_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_eta"),
           "pfCand_neutral_phi_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_phi"),
        "pfCand_neutral_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_dxy_pf"),
         "pfCand_neutral_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_dz_pf"),
   "pfCand_neutral_puppiWeight_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_puppiWeight"),
  "pfCand_neutral_hcalFraction_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_hcalFraction"),
        "pfCand_neutral_fromPV_ptRelSorted":operator.attrgetter( "DL_pfCand_neutral_fromPV"),
        },
                            'charged':{
            "pfCand_charged_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_pt"),
           "pfCand_charged_eta_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_eta"),
           "pfCand_charged_phi_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_phi"),
        "pfCand_charged_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_dxy_pf"),
         "pfCand_charged_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_dz_pf"),
   "pfCand_charged_puppiWeight_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_puppiWeight"),
  "pfCand_charged_hcalFraction_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_hcalFraction"),
        "pfCand_charged_fromPV_ptRelSorted":operator.attrgetter( "DL_pfCand_charged_fromPV"),
        },
                            'photon':{
             "pfCand_photon_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_pt"),
            "pfCand_photon_eta_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_eta"),
            "pfCand_photon_phi_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_phi"),
         "pfCand_photon_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_dxy_pf"),
          "pfCand_photon_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_dz_pf"),
    "pfCand_photon_puppiWeight_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_puppiWeight"),
   "pfCand_photon_hcalFraction_ptRelSorted":operator.attrgetter( "DL_pfCand_photon_hcalFraction"),
        },
                            'electron':{
               "pfCand_electron_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_pt"),
              "pfCand_electron_eta_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_eta"),
              "pfCand_electron_phi_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_phi"),
           "pfCand_electron_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_dxy_pf"),
            "pfCand_electron_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_electron_dz_pf"),
        },
                            'muon':{
               "pfCand_muon_pt_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_pt"),
              "pfCand_muon_eta_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_eta"),
              "pfCand_muon_phi_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_phi"),
           "pfCand_muon_dxy_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_dxy_pf"),
            "pfCand_muon_dz_pf_ptRelSorted":operator.attrgetter( "DL_pfCand_muon_dz_pf"),
        },
    }

    # for a given lepton, read the pf candidate mask and return the PF indices
    def get_pf_indices( self, pf_type, collection_name, n_lep):
        n = getattr(self.event, "nDL_pfCand_%s"%pf_type )
        pf_mask = getattr(self.event, "DL_pfCand_%s_%s_mask" % (pf_type, "selectedLeptons" if collection_name=="LepGood" else "otherLeptons" ) )
        mask_ = (1<<n_lep)
        return filter( lambda i: mask_&pf_mask[i], range(n))

    def getEntry( self, nevent ):
        self.chain.GetEntry( nevent )
        self.nevent = nevent

    def _get_all_pf_candidates( self, flavor):
        n = self.pf_size_getters[flavor](self.event)
        att_getters = self.pf_getters[flavor]
        return [ {name: getter(self.event)[i] for name, getter in self.pf_getters[flavor].iteritems()} for i in range(n) ]
    
    # cached version of get_all_pf_candidates
    @property
    def pf_candidates( self ):
        if self._nevent == self.nevent:
            return self._pf_candidates
        else:
            self._pf_candidates = {flavor: self._get_all_pf_candidates(flavor) for flavor in self.flavors}
            self._nevent = self.nevent 
            return self._pf_candidates

    # put all inputs together
    def pf_candidates_for_lepton( self, collection_name, n_lep):

        # read pf indices, then select the candidates
        pf_candidates = {}
        for flavor in self.flavors:
            pf_indices            = self.get_pf_indices( flavor, collection_name, n_lep )
            pf_candidates[flavor] = [ self.pf_candidates[flavor][i] for i in pf_indices]

            # now calculate the pf_candidate features that depend on the lepton in question
            lep_p4 = ROOT.TLorentzVector()
            lep_getters = self.feature_getters( collection_name )
            lep_p4.SetPtEtaPhiM( lep_getters["lep_pt"](self.event)[n_lep], lep_getters["lep_eta"](self.event)[n_lep], lep_getters["lep_phi"](self.event)[n_lep], 0. )

            name = "pfCand_"+flavor+"_%s_ptRelSorted"
            ptRel_name = name%"ptRel"
            dR_name    = name%"dR"
            for cand in pf_candidates[flavor]:

                cand_p4 = ROOT.TLorentzVector()
                cand_p4.SetPtEtaPhiM( 
                    cand[name%"pt"], cand[name%"eta"],cand[name%"phi"],0.
                    )
                
                cand[ptRel_name] = ptRel( cand_p4, lep_p4 )
                cand[dR_name]    = deltaR( cand[name%"eta"], cand[name%"phi"], lep_getters["lep_eta"](self.event)[n_lep], lep_getters["lep_phi"](self.event)[n_lep])

            # ptRel sorting
            pf_candidates[flavor].sort( key = lambda p:-p[ptRel_name] )
 
        return pf_candidates

    def features_for_lepton( self, collection_name, feature_branches, n_lep):
        # read the lepton features
        return [ self.feature_getters( collection_name )[b](self.event)[n_lep] for b in feature_branches ]

    def prepare_inputs( self, collection_name, feature_branches, pf_branches, n_lep, means):
        
        features      = self.features_for_lepton( collection_name, feature_branches, n_lep )
        pf_candidates = self.pf_candidates_for_lepton( collection_name, n_lep )
         
if __name__ == "__main__": 
    # Information on the training
    training_directory = '/afs/hephy.at/data/gmoertl01/DeepLepton/trainings/muons/20181013/DYVsQCD_ptRelSorted_MuonTrainData'
    trainingInfo = TrainingInfo( training_directory )

    # Input data
    input_filename = "/afs/hephy.at/work/r/rschoefbeck/CMS/tmp/CMSSW_9_4_6_patch1/src/CMGTools/StopsDilepton/cfg/test/WZTo3LNu_amcatnlo_1/treeProducerSusySingleLepton/tree.root"
    inputData = InputData( input_filename )

    inputData.getEntry(0)
    pf_candidates = inputData.pf_candidates_for_lepton("LepGood", 0)
    features      =  inputData.features_for_lepton( "LepGood", trainingInfo.branches[0], 0 )

    #res2 = inputData.read_inputs("LepOther", trainingInfo.branches, 0)

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
