#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.models import Model
import warnings
warnings.warn("DeepJet_models.py is deprecated and will be removed! Please move to the models directory", DeprecationWarning)

from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D
#fix for dropout on gpus

#import tensorflow
#from tensorflow.python.ops import control_flow_ops 
#tensorflow.python.control_flow_ops = control_flow_ops

from TrainDataDeepLepton import TrainData_fullTruth, TrainData_fullTruth_simple, TrainData_fullTruth_prompt
from TrainDataDeepLepton import fileTimeOut

#list of classes defined below:
#class TrainData_deepLeptons_Electrons_2016_run1(TrainData_fullTruth):
    #this class is a copy of class TrainData_deepLeptons_Electrons_2016 (see below) with most of "def _init_" taken from class TrainData_deepLeptons_Muons_sorted_2016, except electron specific lines. This is intended to replace TrainData_deepLeptons_Electrons_2016 once it runs successfully
#class TrainData_deepLeptons_Electrons_2016(TrainData_fullTruth):
#class TrainData_deepLeptons_Muons_sorted_2016(TrainData_fullTruth):
#class TrainData_deepLeptons_Muons_simpleClasses_2016(TrainData_fullTruth_simple):
#class TrainData_deepLeptons_Muons_globalVarsOnly_2016(TrainData_fullTruth):
#class TrainData_deepLeptons_Muons_TTV_2016(TrainData_fullTruth):
#class TrainData_deepLeptons_Muons_TTVonly_2016(TrainData_fullTruth):


class TrainData_deepLeptons_Electrons_2016_run1(TrainData_fullTruth):
    #this class is a copy of class TrainData_deepLeptons_Electrons_2016 (see below) with most of "def _init_" taken from class TrainData_deepLeptons_Muons_sorted_2016, except electron specific lines. This is intended to replace TrainData_deepLeptons_Electrons_2016 once it runs successfully
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)

        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        #'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        'lep_jetPtRatiov2', 'lep_jetPtRelv2',
        #'lep_neutralHadronIsoR03', 'lep_chargedHadronIsoR03',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #electron specific features

        'lep_etaSc', #'lep_sigmaIEtaIEta', 
        'lep_full5x5_sigmaIetaIeta', 
        'lep_dEtaInSeed', 
        'lep_dPhiScTrkIn', 
        'lep_dEtaScTrkIn', 
        'lep_eInvMinusPInv', 
        'lep_convVeto_float', #apply?
        'lep_hadronicOverEm', 'lep_r9', 
        'lep_mvaIdSpring16', #try without this?
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],8) #5
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted', 'pfCand_charged_dz_pf_ptRelSorted', 'pfCand_charged_dzAssociatedPV_ptRelSorted',],50) #25
        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],20) #10
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', #'SV_edxy_ptSorted', 
                           'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted', 'SV_secD3dTracks_ptSorted',],4)


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]


class TrainData_deepLeptons_Electrons_2016(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        # 'lep_jetPtRatiov2', 'lep_jetPtRelv2',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #electron specific features
        'lep_etaSc', 'lep_sigmaIEtaIEta', 'lep_full5x5_sigmaIetaIeta', 
        'lep_dEtaInSeed', 'lep_dPhiScTrkIn', 'lep_dEtaScTrkIn', 
        'lep_eInvMinusPInv', 'lep_convVeto_float', 'lep_hadronicOverEm', 'lep_r9', 
        'lep_mvaIdSpring16',
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],5)
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted', 'pfCand_charged_dz_pf_ptRelSorted', 'pfCand_charged_dzAssociatedPV_ptRelSorted',],25)
        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],10)
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', 'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted', 'SV_secD3dTracks_ptSorted'],4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]

class TrainData_deepLeptons_Muons_sorted_2016(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        #'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        'lep_jetPtRatiov2', 'lep_jetPtRelv2',
        #'lep_neutralHadronIsoR03', 'lep_chargedHadronIsoR03',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #muon specific features
        'lep_segmentCompatibility', 'lep_muonInnerTrkRelErr', 'lep_isGlobalMuon_float', 
        'lep_chi2LocalPosition', 'lep_chi2LocalMomentum', 'lep_globalTrackChi2', 
        'lep_glbTrackProbability', 'lep_trkKink', 'lep_caloCompatibility', 
        'lep_nStations_float', 
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],7) #5
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted', 'pfCand_charged_dz_pf_ptRelSorted', 'pfCand_charged_dzAssociatedPV_ptRelSorted',],50) #25
        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],20) #10
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', #'SV_edxy_ptSorted', 
                           'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted', 'SV_secD3dTracks_ptSorted',],4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]


class TrainData_deepLeptons_Muons_simpleClasses_2016(TrainData_fullTruth_simple):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth_simple.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        'lep_jetPtRatiov2', 'lep_jetPtRelv2',
        #'lep_neutralHadronIsoR03', 'lep_chargedHadronIsoR03',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #muon specific features
        'lep_segmentCompatibility', 'lep_muonInnerTrkRelErr', 'lep_isGlobalMuon_float', 
        'lep_chi2LocalPosition', 'lep_chi2LocalMomentum', 'lep_globalTrackChi2', 
        'lep_glbTrackProbability', 'lep_trkKink', 'lep_caloCompatibility', 
        'lep_nStations_float', 
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],5)
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted', 'pfCand_charged_dz_pf_ptRelSorted', 'pfCand_charged_dzAssociatedPV_ptRelSorted',],25)
        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],10)
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', 'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted', 'SV_secD3dTracks_ptSorted'],4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]



class TrainData_deepLeptons_Muons_globalVarsOnly_2016(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        'lep_jetPtRatiov2', 'lep_jetPtRelv2',
        #'lep_neutralHadronIsoR03', 'lep_chargedHadronIsoR03',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #muon specific features
        'lep_segmentCompatibility', 'lep_muonInnerTrkRelErr', 'lep_isGlobalMuon_float', 
        'lep_chi2LocalPosition', 'lep_chi2LocalMomentum', 'lep_globalTrackChi2', 
        'lep_glbTrackProbability', 'lep_trkKink', 'lep_caloCompatibility', 
        'lep_nStations_float', 
        ])

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)

        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global]
        self.y=[alltruth]


class TrainData_deepLeptons_Muons_TTV_2016(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', 
        'lep_dxy', 'lep_dz', 'lep_sip3d',  

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        'lep_jetPtRatiov2', 'lep_jetPtRelv2', 'lep_jetDR', 'lep_jetBTagDeepCSV', 

        #muon specific features
        'lep_segmentCompatibility', 
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],5)
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted', 'pfCand_charged_dz_pf_ptRelSorted', 'pfCand_charged_dzAssociatedPV_ptRelSorted',],25)
        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],10)
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', 'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted', 'SV_secD3dTracks_ptSorted'],4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]

class TrainData_deepLeptons_Muons_TTVonly_2016(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', 
        'lep_dxy', 'lep_dz', 'lep_sip3d',  

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        'lep_jetPtRatiov2', 'lep_jetPtRelv2', 'lep_jetDR', 'lep_jetBTagDeepCSV', 

        #muon specific features
        'lep_segmentCompatibility', 
        ])

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)

        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global]
        self.y=[alltruth]


class TrainData_deepLeptons_Muons_2017(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)


        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        #'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        'lep_jetPtRatiov2', 'lep_jetPtRelv2',
        #'lep_neutralHadronIsoR03', 'lep_chargedHadronIsoR03',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #muon specific features
        'lep_segmentCompatibility', 'lep_muonInnerTrkRelErr', 'lep_isGlobalMuon_float', 
        'lep_chi2LocalPosition', 'lep_chi2LocalMomentum', 'lep_globalTrackChi2', 
        'lep_glbTrackProbability', 'lep_trkKink', 'lep_caloCompatibility', 
        'lep_nStations_float', 
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],5) 
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted',  'pfCand_charged_dz_pf_ptRelSorted_patch', 'pfCand_charged_dzAssociatedPV_ptRelSorted_patch',],50)

        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],10)
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', #'SV_edxy_ptSorted', 
                           'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted_patch', 
                           'SV_secD3dTracks_ptSorted_patch',],4)

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]


class TrainData_deepLeptons_Electrons_2017_run1(TrainData_fullTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth.__init__(self)

        self.addBranches([
        #global lepton features
        'lep_pt', 'lep_eta', #'lep_phi', 
        'lep_dxy', 'lep_dz', #'lep_edxy', 'lep_edz', 
        'lep_ip3d', 'lep_sip3d',  
        'lep_innerTrackChi2','lep_innerTrackValidHitFraction',
        'lep_ptErrTk', 'lep_rho', 'lep_jetDR',
        'lep_trackerLayers_float', 'lep_pixelLayers_float', 'lep_trackerHits_float', 'lep_lostHits_float', 'lep_lostOuterHits_float',

        #isolation features
        'lep_relIso03', 'lep_miniRelIsoCharged', 'lep_miniRelIsoNeutral', 
        #'lep_jetPtRatiov1', 'lep_jetPtRelv1',
        'lep_jetPtRatiov2', 'lep_jetPtRelv2',
        #'lep_neutralHadronIsoR03', 'lep_chargedHadronIsoR03',

        #high-level lepton features
        #lep_jetBTagCSV', 
        'lep_jetBTagDeepCSV', 
        #'lep_jetBTagDeepCSVCvsB', 'lep_jetBTagDeepCSVCvsL',  

        #electron specific features

        'lep_etaSc', #'lep_sigmaIEtaIEta', 
        'lep_full5x5_sigmaIetaIeta', 
        'lep_dEtaInSeed', 
        'lep_dPhiScTrkIn', 
        'lep_dEtaScTrkIn', 
        'lep_eInvMinusPInv', 
        'lep_convVeto_float', #apply?
        'lep_hadronicOverEm', 'lep_r9', 
        'lep_mvaIdSpring16', #try without this?
        ])

        self.addBranches(['pfCand_neutral_ptRel_ptRelSorted',  'pfCand_neutral_deltaR_ptRelSorted',  'pfCand_neutral_pt_ptRelSorted',  'pfCand_neutral_puppiWeight_ptRelSorted', 'pfCand_neutral_fromPV_ptRelSorted',],5)
        self.addBranches(['pfCand_charged_ptRel_ptRelSorted',  'pfCand_charged_deltaR_ptRelSorted',  'pfCand_charged_pt_ptRelSorted',  'pfCand_charged_puppiWeight_ptRelSorted', 'pfCand_charged_fromPV_ptRelSorted', 'pfCand_charged_dxy_pf_ptRelSorted',     'pfCand_charged_dz_pf_ptRelSorted_patch', 'pfCand_charged_dzAssociatedPV_ptRelSorted_patch',],25)
        self.addBranches(['pfCand_photon_ptRel_ptRelSorted',   'pfCand_photon_deltaR_ptRelSorted',   'pfCand_photon_pt_ptRelSorted',   'pfCand_photon_puppiWeight_ptRelSorted',  'pfCand_photon_fromPV_ptRelSorted',],10)
        self.addBranches(['pfCand_electron_ptRel_ptRelSorted', 'pfCand_electron_deltaR_ptRelSorted', 'pfCand_electron_pt_ptRelSorted', 'pfCand_electron_dxy_pf_ptRelSorted', 'pfCand_electron_dz_pf_ptRelSorted',],3)
        self.addBranches(['pfCand_muon_ptRel_ptRelSorted',     'pfCand_muon_deltaR_ptRelSorted',     'pfCand_muon_pt_ptRelSorted',     'pfCand_muon_dxy_pf_ptRelSorted',     'pfCand_muon_dz_pf_ptRelSorted',    ],3)
        self.addBranches([ 'SV_pt_ptSorted', 'SV_chi2_ptSorted', 'SV_ndof_ptSorted', 'SV_dxy_ptSorted', #'SV_edxy_ptSorted', 
                           'SV_ip3d_ptSorted', 'SV_eip3d_ptSorted', 'SV_sip3d_ptSorted', 
                           'SV_cosTheta_ptSorted', 'SV_deltaR_ptSorted', 'SV_maxDxyTracks_ptSorted', 'SV_secDxyTracks_ptSorted', 'SV_maxD3dTracks_ptSorted_patch', 'SV_secD3dTracks_ptSorted_patch',],4)


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch

        sw=stopwatch()
        swall=stopwatch()

        import ROOT
        
        self.treename="tree"
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')

        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        x_ppf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        x_epf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[4],
                                   self.branchcutoffs[4],self.nsamples)
        x_mpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[5],
                                   self.branchcutoffs[5],self.nsamples)
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[6],
                                   self.branchcutoffs[6],self.nsamples)


        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')

        Tuple = self.readTreeFromRootToTuple(filename)

        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple['lep_isMuon']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)


        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)

        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_ppf=x_ppf[notremoves > 0]
            x_epf=x_epf[notremoves > 0]
            x_mpf=x_mpf[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        print(x_global.shape,self.nsamples)

        self.w=[weights, weights]
        self.x=[x_global,x_npf, x_cpf, x_ppf, x_epf, x_mpf, x_sv]
        self.y=[alltruth]
