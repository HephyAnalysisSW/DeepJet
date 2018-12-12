from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM,merge, Convolution1D, Conv2D, Bidirectional, MaxPooling1D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Multiply
from buildingBlocks_deepLepton import block_deepLeptonConvolutions, block_deepLeptonDense, block_deepLeptonConvolutions_pooling, block_deepLeptonConvolutions_testSize, block_deepLeptonDense_testSize, block_deepLeptonDense_testSplit_sum, block_deepLeptonDense_testSplit_cands, block_deepLeptonDense_testSplit_global #, block_SchwartzImage, block_deepFlavourBTVConvolutions

def model_deepLeptonReference(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):
    """
    reference 1x1 convolutional model for 'deepLepton'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2])
    ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3])
    epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4])
    mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6])
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    npf, cpf, ppf, epf, mpf, vtx = block_deepLeptonConvolutions(neutrals=npf,
                                                charged=cpf,
                                                photons=ppf,
                                                electrons=epf,
                                                muons=mpf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    ppf = LSTM(50,go_backwards=True,implementation=2, name='ppf_lstm')(ppf)
    ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
    ppf = Dropout(dropoutRate)(ppf)
    
    epf = LSTM(50,go_backwards=True,implementation=2, name='epf_lstm')(epf)
    epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
    epf = Dropout(dropoutRate)(epf)
    
    mpf = LSTM(50,go_backwards=True,implementation=2, name='mpf_lstm')(mpf)
    mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
    mpf = Dropout(dropoutRate)(mpf)
    
    vtx = LSTM(150,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,npf,cpf,ppf,epf,mpf,vtx])
    
    x = block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    #reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    #reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [lepton_pred]
    #predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    #model.save("/local/gmoertl/DeepLepton/DeepJet_GPU/DeepJet/KERAS_initial_model.h5")
    return model

def model_deepLeptonReference_biLSTM(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):
    """
    reference 1x1 convolutional model for 'deepLepton'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2])
    ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3])
    epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4])
    mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6])
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    npf, cpf, ppf, epf, mpf, vtx = block_deepLeptonConvolutions(neutrals=npf,
                                                charged=cpf,
                                                photons=ppf,
                                                electrons=epf,
                                                muons=mpf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    npf = Bidirectional(LSTM(50,implementation=2, name='npf_lstm'), merge_mode='concat')(npf)
    npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    cpf = Bidirectional(LSTM(150,implementation=2, name='cpf_lstm'), merge_mode='concat')(cpf)
    cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    ppf = Bidirectional(LSTM(50,implementation=2, name='ppf_lstm'), merge_mode='concat')(ppf)
    ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
    ppf = Dropout(dropoutRate)(ppf)
    
    epf = Bidirectional(LSTM(50,implementation=2, name='epf_lstm'), merge_mode='concat')(epf)
    epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
    epf = Dropout(dropoutRate)(epf)
    
    mpf = Bidirectional(LSTM(50,implementation=2, name='mpf_lstm'), merge_mode='concat')(mpf)
    mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
    mpf = Dropout(dropoutRate)(mpf)
    
    vtx = Bidirectional(LSTM(150,implementation=2, name='vtx_lstm'), merge_mode='concat')(vtx)
    vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,npf,cpf,ppf,epf,mpf,vtx])
    
    x = block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    #reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    #reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [lepton_pred]
    #predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    #model.save("/local/gmoertl/DeepLepton/DeepJet_GPU/DeepJet/KERAS_initial_model.h5")
    return model

def model_deepLeptonReference_pooling(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):
    """
    reference 1x1 convolutional model for 'deepLepton'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2])
    ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3])
    epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4])
    mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6])
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    npf, cpf, ppf, epf, mpf, vtx = block_deepLeptonConvolutions_pooling(neutrals=npf,
                                                charged=cpf,
                                                photons=ppf,
                                                electrons=epf,
                                                muons=mpf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    ppf = LSTM(50,go_backwards=True,implementation=2, name='ppf_lstm')(ppf)
    ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
    ppf = Dropout(dropoutRate)(ppf)
    
    epf = LSTM(50,go_backwards=True,implementation=2, name='epf_lstm')(epf)
    epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
    epf = Dropout(dropoutRate)(epf)
    
    mpf = LSTM(50,go_backwards=True,implementation=2, name='mpf_lstm')(mpf)
    mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
    mpf = Dropout(dropoutRate)(mpf)
    
    vtx = LSTM(150,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,npf,cpf,ppf,epf,mpf,vtx])
    
    x = block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    #reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    #reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [lepton_pred]
    #predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    #model.save("/local/gmoertl/DeepLepton/DeepJet_GPU/DeepJet/KERAS_initial_model.h5")
    return model



def model_deepLeptonReference_testSize(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):
    """
    reference 1x1 convolutional model for 'deepLepton'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2])
    ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3])
    epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4])
    mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6])
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    npf, cpf, ppf, epf, mpf, vtx = block_deepLeptonConvolutions_testSize(neutrals=npf,
                                                charged=cpf,
                                                photons=ppf,
                                                electrons=epf,
                                                muons=mpf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    ppf = LSTM(50,go_backwards=True,implementation=2, name='ppf_lstm')(ppf)
    ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
    ppf = Dropout(dropoutRate)(ppf)
    
    epf = LSTM(50,go_backwards=True,implementation=2, name='epf_lstm')(epf)
    epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
    epf = Dropout(dropoutRate)(epf)
    
    mpf = LSTM(50,go_backwards=True,implementation=2, name='mpf_lstm')(mpf)
    mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
    mpf = Dropout(dropoutRate)(mpf)
    
    vtx = LSTM(150,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,npf,cpf,ppf,epf,mpf,vtx])
    
    x = block_deepLeptonDense_testSize(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    #reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    #reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [lepton_pred]
    #predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    #model.save("/local/gmoertl/DeepLepton/DeepJet_GPU/DeepJet/KERAS_initial_model.h5")
    return model



def model_deepLeptonReference_testSplit(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):
    """
    reference 1x1 convolutional model for 'deepLepton'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2])
    ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3])
    epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4])
    mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6])
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    npf, cpf, ppf, epf, mpf, vtx = block_deepLeptonConvolutions(neutrals=npf,
                                                charged=cpf,
                                                photons=ppf,
                                                electrons=epf,
                                                muons=mpf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    ppf = LSTM(50,go_backwards=True,implementation=2, name='ppf_lstm')(ppf)
    ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
    ppf = Dropout(dropoutRate)(ppf)
    
    epf = LSTM(50,go_backwards=True,implementation=2, name='epf_lstm')(epf)
    epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
    epf = Dropout(dropoutRate)(epf)
    
    mpf = LSTM(50,go_backwards=True,implementation=2, name='mpf_lstm')(mpf)
    mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
    mpf = Dropout(dropoutRate)(mpf)
    
    vtx = LSTM(150,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    #separate DNN for pfCands+SV and global vars
    xCands  = Concatenate()( [npf,cpf,ppf,epf,mpf,vtx])
    xGlobal = globalvars
    
    xCands  = block_deepLeptonDense_testSplit_cands(xCands,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    xGlobal = block_deepLeptonDense_testSplit_global(xGlobal,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    x       = Concatenate()( [xGlobal,xCands])
    x       = block_deepLeptonDense_testSplit_sum(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)

    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    #reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    #reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [lepton_pred]
    #predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    #model.save("/local/gmoertl/DeepLepton/DeepJet_GPU/DeepJet/KERAS_initial_model.h5")
    return model

def model_deepLeptonReference_globalVarsOnly(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):
    """
    reference 1x1 convolutional model for 'deepLepton'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    
    #x = Concatenate()( [globalvars,npf,cpf,ppf,epf,mpf,vtx])
    x = globalvars
    
    x = block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [lepton_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepLeptonReference_noCNN(Inputs,nclasses,nregclasses,dropoutRate=0.5,momentum=0.2):

    #Batch Normalization
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[1])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[2])
    ppf    =     BatchNormalization(momentum=momentum,name='ppf_input_batchnorm')     (Inputs[3])
    epf    =     BatchNormalization(momentum=momentum,name='epf_input_batchnorm')     (Inputs[4])
    mpf    =     BatchNormalization(momentum=momentum,name='mpf_input_batchnorm')     (Inputs[5])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[6])
    
    #LSTMs
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    ppf = LSTM(50,go_backwards=True,implementation=2, name='ppf_lstm')(ppf)
    ppf = BatchNormalization(momentum=momentum,name='ppflstm_batchnorm')(ppf)
    ppf = Dropout(dropoutRate)(ppf)
    
    epf = LSTM(50,go_backwards=True,implementation=2, name='epf_lstm')(epf)
    epf = BatchNormalization(momentum=momentum,name='epflstm_batchnorm')(epf)
    epf = Dropout(dropoutRate)(epf)
    
    mpf = LSTM(50,go_backwards=True,implementation=2, name='mpf_lstm')(mpf)
    mpf = BatchNormalization(momentum=momentum,name='mpflstm_batchnorm')(mpf)
    mpf = Dropout(dropoutRate)(mpf)
    
    vtx = LSTM(150,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    #Dense Neural Network
    x = Concatenate()( [globalvars,npf,cpf,ppf,epf,mpf,vtx])
    x = block_deepLeptonDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)

    #Output
    lepton_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    predictions = [lepton_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model



