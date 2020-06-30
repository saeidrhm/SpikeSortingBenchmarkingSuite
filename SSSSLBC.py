from SSANNCommon import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, UpSampling1D, Dropout, Reshape
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import seed
import os
import random

def AutoEncoder_CNN_enc_dec_output(CNNStructure, TrainParams, LayerArgList,img_rows):
    print("AutoEncoder_CNN_enc_dec_output::CNNStructure: "+str(CNNStructure))
    count = 0
    NLayers = len(LayerArgList)
    img_rows_shape_in_middle = img_rows
    for i in range(0,int(NLayers)-1):
        img_rows_shape_in_middle/= LayerArgList[i]["CNN_MaxPooling_Size"]
    print("img_rows_shape_in_middle: "+str(img_rows_shape_in_middle))
    input_img = Input(shape=(img_rows,1))
    print("input_img: "+str(input_img))
    OutLayer = Conv1D(filters = int(CNNStructure[0]),kernel_size=int(LayerArgList[count]["CNN_Filter_Size"]),strides = int(LayerArgList[count]["CNN_Stride"]), activation=LayerArgList[count]['Dense_activation'],padding='same')(input_img)
    #OutLayer = Conv1D(32,5,padding= 'same')(input_img)
    OutLayer = BatchNormalization()(OutLayer)
    OutLayer = Dropout(rate=float(LayerArgList[count]['Dropout_rate']))(OutLayer)
    OutLayer = MaxPooling1D(LayerArgList[count]["CNN_MaxPooling_Size"])(OutLayer)
    #encoded = OutLayer
    if NLayers>= 3:
        for count in range(1,int(NLayers-1)):
            OutLayer = Conv1D( filters = CNNStructure[count],kernel_size=LayerArgList[count]["CNN_Filter_Size"],strides = LayerArgList[count]["CNN_Stride"], activation=LayerArgList[0]['Dense_activation'],padding='same')(OutLayer) 
            OutLayer = BatchNormalization()(OutLayer)
            OutLayer = Dropout(rate=float(LayerArgList[count]['Dropout_rate']))(OutLayer)
            OutLayer = MaxPooling1D(LayerArgList[count]["CNN_MaxPooling_Size"])(OutLayer)
    if NLayers>= 2:
        CNNInputshape = [int(img_rows_shape_in_middle), int(CNNStructure[-2])]
        print("CNNInputshape: "+str(CNNInputshape))
        OutLayer = Flatten()(OutLayer)
        OutLayer = Dense(CNNStructure[-1], activation=LayerArgList[0]['Dense_activation'])(OutLayer)
        encoded  = BatchNormalization(name="encoded_vals")(OutLayer)
        OutLayer = encoded
        OutLayer = Dropout(rate=float(LayerArgList[-1]['Dropout_rate']))(OutLayer)
        OutLayer = Dense(int(CNNInputshape[0]*CNNInputshape[1]), activation=LayerArgList[count]['Dense_activation'])(OutLayer)
        OutLayer = BatchNormalization()(OutLayer)
        OutLayer = Dropout(rate=float(LayerArgList[-1]['Dropout_rate']))(OutLayer)
        OutLayer = Reshape(CNNInputshape)(OutLayer)
    if NLayers>= 2:
        for count in range(int(NLayers)-2,-1,-1):
            OutLayer = Conv1D( filters = CNNStructure[count],kernel_size=LayerArgList[count]["CNN_Filter_Size"],strides = LayerArgList[count]["CNN_Stride"], activation=LayerArgList[0]['Dense_activation'],padding='same')(OutLayer)
            OutLayer = BatchNormalization()(OutLayer)
            OutLayer = Dropout(rate=float(LayerArgList[count]['Dropout_rate']))(OutLayer)
            OutLayer = UpSampling1D(LayerArgList[count]["CNN_MaxPooling_Size"])(OutLayer)
        
    decoded = Conv1D( filters = 1,kernel_size=LayerArgList[0]["CNN_Filter_Size"],strides = LayerArgList[0]["CNN_Stride"], activation=TrainParams['Autoeoncoder_lastlayer_activation'],padding='same',name="decoded_vals")(OutLayer)

    autoencoder_model = Model(inputs=input_img, outputs= [encoded, decoded])
    return(autoencoder_model,input_img,encoded,decoded)




def AutoEncoder_MLP_enc_dec_output(MLPStructure, TrainParams, LayerArgList,img_rows):
    count = 0
    print("MLPStructure: "+str(MLPStructure))
    print("LayerArgList :"+str(LayerArgList))
    NLayers = len(LayerArgList)
    input_img = Input(shape=(img_rows,))
    OutLayer = Dense(MLPStructure[0], activation=LayerArgList[0]['Dense_activation'])(input_img)
    OutLayer = BatchNormalization()(OutLayer)
    OutLayer = Dropout(rate=float(LayerArgList[0]['Dropout_rate']))(OutLayer)
    if NLayers>= 3:
        for count in range(1,int(NLayers-1)):
            #print("count: "+str(count))
            #print("LayerArgList[count]: "+str(LayerArgList[count]))
            #print("activation=LayerArgList[count]['Dense_activation']: "+str(LayerArgList[count]['Dense_activation']))
            OutLayer = Dense(MLPStructure[count], activation=LayerArgList[count]['Dense_activation'])(OutLayer)
            OutLayer = BatchNormalization()(OutLayer)
            OutLayer = Dropout(rate=float(LayerArgList[count]['Dropout_rate']))(OutLayer)

    if NLayers>= 2:
        OutLayer = Dense(MLPStructure[-1], activation=LayerArgList[-1]['Dense_activation'])(OutLayer)
        encoded  = BatchNormalization(name="encoded_vals")(OutLayer)
        OutLayer = encoded
        OutLayer = Dropout(rate=float(LayerArgList[-1]['Dropout_rate']))(OutLayer)

    if NLayers>= 2:
        for count in range(int(NLayers)-2,-1,-1):
            OutLayer = Dense(MLPStructure[count], activation=LayerArgList[count]['Dense_activation'])(OutLayer)
            OutLayer = BatchNormalization()(OutLayer)
            OutLayer = Dropout(rate=float(LayerArgList[count]['Dropout_rate']))(OutLayer)

    decoded = Dense(img_rows, activation=TrainParams['Autoeoncoder_lastlayer_activation'],name="decoded_vals")(OutLayer)
    autoencoder_model = Model(inputs=input_img, outputs=[encoded, decoded])
    return(autoencoder_model,input_img,encoded,decoded)



def SSLBetweenChanFeatureExtractionChan(x_train_chan_0,x_train_chan_1,Parameters):
    SpikeLen = Parameters['SSLBC_SpikeLen'] ##128
    num_classes_combined = 2
    input_dim = SpikeLen
    LossWeghts = Parameters['SSLBC_LossWeghts']  ##[1, 0.5, 0.5]
    Validation_Precent  = Parameters['SSLBC_Validation_Precent']
    GNoiseSD = Parameters['SSLBC_GNoiseSD']
    CNNStructure = Parameters["SSLBC_CNNStructure"]
    MLPStructure = Parameters["SSLBC_MLPStructure"]
    TrainParams = Parameters["SSLBC_TrainParams"]
    LayerArgList = Parameters["SSLBC_LayerArgList"]
    epochs = TrainParams['epochs']
    if Parameters['SSLBC_Normaliztion']==True :
        TrainParams["Autoeoncoder_lastlayer_activation"] = "tanh"
    else:
        TrainParams["Autoeoncoder_lastlayer_activation"] = "linear"
    y_train_chan_0 = np.zeros(x_train_chan_0.shape[0])
    y_train_chan_1 = np.ones(x_train_chan_0.shape[0])
    if Parameters["SSLBC_CNNModel"]==True : 
        # Data 
        if K.image_data_format() == 'channels_first':
            x_train_chan_0 = x_train_chan_0.reshape(x_train_chan_0.shape[0], 1, input_dim)
            x_train_chan_1 = x_train_chan_1.reshape(x_train_chan_1.shape[0], 1, input_dim)
            input_shape = Input(shape=(1, input_dim))
        else:
             x_train_chan_0 = x_train_chan_0.reshape(x_train_chan_0.shape[0], input_dim, 1)
             x_train_chan_1 = x_train_chan_1.reshape(x_train_chan_1.shape[0], input_dim, 1)
             input_shape = Input(shape=(input_dim, 1))
        if(len(x_train_chan_0)>len(x_train_chan_1)):
          rep = True
        else:
          rep = False
        x_train_chan_1_selected_idx = np.random.choice(len(x_train_chan_1),len(x_train_chan_0), replace=rep)
        x_train_chan_1_selected = x_train_chan_1[x_train_chan_1_selected_idx,:,:]
        x_train_chan_0_prime = x_train_chan_0
        random_noise =  np.random.normal(0,GNoiseSD,(int)(x_train_chan_0.shape[0]*x_train_chan_0.shape[1]*x_train_chan_0.shape[2]))*np.ptp(x_train_chan_0)
        random_noise = random_noise.reshape(((int)(x_train_chan_0.shape[0]),x_train_chan_0.shape[1],x_train_chan_0.shape[2]))
        x_train_chan_0_prime[range(0,(int)(x_train_chan_0.shape[0])),:,:] = x_train_chan_0[range(0,(int)(x_train_chan_0.shape[0])),:,:]+random_noise
        x_train_chan_1_selected_prime_idx = np.random.choice(len(x_train_chan_0),len(x_train_chan_0), replace=False)
        x_train_chan_1_selected_prime = x_train_chan_0[x_train_chan_1_selected_prime_idx,:,:]
        x_train = np.concatenate((x_train_chan_0,x_train_chan_1_selected), axis=0)
        x_test = x_train
        y_train = np.concatenate((y_train_chan_0,y_train_chan_1))
        y_test = y_train
        x_train_prime = np.concatenate((x_train_chan_0_prime,x_train_chan_1_selected_prime), axis=0)
        if Parameters['SSLBC_Normaliztion'] == True:
            x_train_orig = np.copy(x_train_chan_0)/ np.max(np.abs(x_train_chan_0))
        else:
            x_train_orig = np.copy(x_train_chan_0)
        random_idx_test = random.sample(range(len(y_train)), int(Validation_Precent*len(y_train)))
        y_test = y_train[random_idx_test]
        if Parameters['SSLBC_Normaliztion'] == True:
            x_test = x_train[random_idx_test,:,:]/ np.max(np.abs(x_train_chan_0))
            x_test_prime = x_train_prime[random_idx_test,:,:]/ np.max(np.abs(x_train_chan_0))
            x_train = np.delete(x_train, random_idx_test, 0)/ np.max(np.abs(x_train_chan_0))
            x_train_prime = np.delete(x_train_prime, random_idx_test, 0)/ np.max(np.abs(x_train_chan_0))
        else:
            x_test = x_train[random_idx_test,:,:]
            x_test_prime = x_train_prime[random_idx_test,:,:]
            x_train = np.delete(x_train, random_idx_test, 0)
            x_train_prime = np.delete(x_train_prime, random_idx_test, 0)
    else:
        # Data 
        x_train_chan_0 = x_train_chan_0.reshape(x_train_chan_0.shape[0], input_dim).astype('float32')
        x_train_chan_1 = x_train_chan_1.reshape(x_train_chan_1.shape[0], input_dim).astype('float32')
        input_shape = Input(shape=(input_dim))
        if(len(x_train_chan_0)>len(x_train_chan_1)):
          rep = True
        else:
          rep = False
        x_train_chan_1_selected_idx = np.random.choice(len(x_train_chan_1),len(x_train_chan_0), replace=rep)
        x_train_chan_1_selected = x_train_chan_1[x_train_chan_1_selected_idx,:]
        x_train_chan_0_prime = x_train_chan_0
        random_noise =  np.random.normal(0,GNoiseSD,(int)(x_train_chan_0.shape[0]*x_train_chan_0.shape[1]))*np.ptp(x_train_chan_0)
        random_noise = random_noise.reshape(((int)(x_train_chan_0.shape[0]),x_train_chan_0.shape[1]))
        x_train_chan_0_prime[range(0,(int)(x_train_chan_0.shape[0])),:] = x_train_chan_0[range(0,(int)(x_train_chan_0.shape[0])),:]+random_noise
        x_train_chan_1_selected_prime_idx = np.random.choice(len(x_train_chan_0),len(x_train_chan_0), replace=False)
        x_train_chan_1_selected_prime = x_train_chan_0[x_train_chan_1_selected_prime_idx,:]
        x_train = np.concatenate((x_train_chan_0,x_train_chan_1_selected), axis=0)
        x_test = x_train
        y_train = np.concatenate((y_train_chan_0,y_train_chan_1))
        y_test = y_train
        x_train_prime = np.concatenate((x_train_chan_0_prime,x_train_chan_1_selected_prime), axis=0)
        if Parameters['SSLBC_Normaliztion'] == True:
            x_train_orig = np.copy(x_train_chan_0)/ np.max(np.abs(x_train_chan_0))
        else:
            x_train_orig = np.copy(x_train_chan_0)
        random_idx_test = random.sample(range(len(y_train)), int(Validation_Precent*len(y_train)))
        y_test = y_train[random_idx_test]
        if Parameters['SSLBC_Normaliztion'] == True:
            x_test = x_train[random_idx_test,:]/ np.max(np.abs(x_train_chan_0))
            x_test_prime = x_train_prime[random_idx_test,:]/ np.max(np.abs(x_train_chan_0))
            x_train = np.delete(x_train, random_idx_test, 0)/ np.max(np.abs(x_train_chan_0))
            x_train_prime = np.delete(x_train_prime, random_idx_test, 0)/ np.max(np.abs(x_train_chan_0))
        else:
            x_test = x_train[random_idx_test,:]
            x_test_prime = x_train_prime[random_idx_test,:]
            x_train = np.delete(x_train, random_idx_test, 0)
            x_train_prime = np.delete(x_train_prime, random_idx_test, 0) 
    y_train = np.delete(y_train, random_idx_test, 0)
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes_combined)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes_combined)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if Parameters['SSLBC_CNNModel']==True:
        autoencoder_model,input_img,encoded,decoded = AutoEncoder_CNN_enc_dec_output(CNNStructure, TrainParams, LayerArgList, input_dim)
    else:
        autoencoder_model,input_img,encoded,decoded = AutoEncoder_MLP_enc_dec_output(MLPStructure, TrainParams, LayerArgList, input_dim)
    initial_epoch=0
    if Parameters['SSLBC_LoadModel']==True:
        print("Loading Model")
        loaded_model = load_model(Parameters['SSLBC_LoadModelFilename'])
        autoencoder_model.set_weights(loaded_model.get_weights())
        initial_epoch=Parameters['AutoEncoder_TrainParams']['epochs']
        epochs = Parameters['SSLBC_TrainParams']['epochs'] + Parameters['AutoEncoder_TrainParams']['epochs']
    ## source:https://keras.io/getting-started/functional-api-guide/
    # Then define the tell-digits-apart model
    if Parameters["SSLBC_CNNModel"]==True :
        image_a = Input(shape=(input_dim, 1))
        image_b = Input(shape=(input_dim, 1))
    else:
        image_a = Input(shape=(input_dim))
        image_b = Input(shape=(input_dim))
    # The vision model will be shared, weights and all
    [encoded_a, decoded_a] = autoencoder_model(image_a)
    [encoded_b, decoded_b]  = autoencoder_model(image_b)
    # Classification
    concatenated = tensorflow.keras.layers.concatenate([encoded_a, encoded_b])
    flatten = Flatten()(concatenated)
    fc = Dense(int(Parameters['SSLBC_ClassLayerSize']), activation=Parameters['SSLBC_ClassLayerActivation'])(flatten)
    out = Dense(num_classes_combined, activation='softmax', name='classification')(fc)
    model = Model(inputs=[image_a,image_b], outputs=[out, decoded_a,decoded_b])
    model.compile(loss=['categorical_crossentropy', 
                    'mse',
                    'mse'],
                  optimizer=TrainParams['Optimizer'],
                  metrics={'classification': 'accuracy'},
                  loss_weights=LossWeghts)

    print(model.summary())
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=Parameters["SSLBC_EarlyStopping_patience"]),GetBest(monitor='val_loss', verbose=1, mode='min')]
    model.fit([x_train,x_train_prime], 
          [y_train, x_train,x_train_prime],
          batch_size=TrainParams['batch_size'],
          epochs=epochs,
          validation_data= ([x_test,x_test_prime], [ y_test, x_test,x_test_prime]),
          shuffle=True,
          initial_epoch=initial_epoch,
          callbacks =callbacks,
          verbose=1)

    layer_name = 'encoded_vals'
    intermediate_layer_model = Model(inputs=autoencoder_model.get_input_at(0) ,
                                 outputs=autoencoder_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_train_orig[:,:])

    # Testing
    #test = np.random.random(input_shape)[np.newaxis,...]
    if Parameters["SSLBC_CNNModel"]==True :
        NumofFeatures = CNNStructure[-1]
    else:
        NumofFeatures = MLPStructure[-1]
    Features = np.empty([len(x_train_orig[:,:]), NumofFeatures])
    for idx in range(0,len(x_train_orig[:,:])):
        Features[idx,:] = intermediate_output[idx,:]
    return (Features[range(0,x_train_chan_0.shape[0]),:])

def SSLBetweenChanFeatureExtraction(Data,Parametrs):
    RetList = []
    for CurrChan in range(0,len(Data)):
        Parametrs["SSLBC_LoadModelFilename"] = CreateBasicAutoPreTrainedFileName(Parametrs,CurrChan)
        RetList.append(SSLBetweenChanFeatureExtractionChan(Data[CurrChan][:,:],Data[(CurrChan+1)%len(Data)][:,:],Parametrs))
    return(RetList)
