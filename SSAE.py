import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, UpSampling1D, Dropout, Reshape
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split

def AutoEncoder_MLP(MLPStructure, TrainParams, LayerArgList,img_rows):
    print("AutoEncoder_MLP::MLPStructure: "+str(MLPStructure))
    count = 0
    NLayers = len(LayerArgList)
    input_img = Input(shape=(img_rows,))
    OutLayer = Dense(MLPStructure[0], activation=LayerArgList[0]['Dense_activation'])(input_img)
    OutLayer = BatchNormalization()(OutLayer)
    OutLayer = Dropout(rate=float(LayerArgList[0]['Dropout_rate']))(OutLayer)
    if NLayers>= 3:
        for count in range(1,int(NLayers-1)):
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
    autoencoder_model = Model(inputs=input_img, outputs= decoded)
    return(autoencoder_model,input_img,encoded,decoded)




def AutoEncoder_CNN(CNNStructure, TrainParams, LayerArgList,img_rows):
    print("AutoEncoder_CNN::CNNStructure: "+str(CNNStructure))
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
    autoencoder_model = Model(inputs=input_img, outputs= decoded)
    return(autoencoder_model,input_img,encoded,decoded)


def AutoEncoderTrainChan(X, Parametrs):
    MLPStructure = Parametrs["AutoEncoder_MLPStructure"]
    CNNStructure = Parametrs["AutoEncoder_CNNStructure"]
    TrainParams = Parametrs["AutoEncoder_TrainParams"]
    LayerArgList = Parametrs["AutoEncoder_LayerArgList"]
    CNNModel = Parametrs["AutoEncoder_CNNModel"]
    GaussianNoiseStD = Parametrs["AutoEncoder_GaussianNoiseStD"]
    Normaliztion = Parametrs["AutoEncoder_Normaliztion"]
    FeaturesFromNoisyData = Parametrs["AutoEncoder_FeaturesFromNoisyData"]
    AutoEncoder_SavedModelFilename =  Parametrs["AutoEncoder_SavedModelFilename"]
    AutoEncoder_ChanNum =  Parametrs["AutoEncoder_ChanNum"]
    AutoEncoder_LoadModelFilename = Parametrs["AutoEncoder_LoadModelFilename"]
    AutoEncoder_LoadModel = Parametrs["AutoEncoder_LoadModel"]
    epochs=TrainParams["epochs"]
    tensorflow.random.set_seed(Parametrs["AutoEncoder_Seed"])
    seed(Parametrs["AutoEncoder_Seed"])
    os.environ['PYTHONHASHSEED']=str(Parametrs["AutoEncoder_Seed"])
    random.seed(Parametrs["AutoEncoder_Seed"])
    (x_train, x_test) = train_test_split(X, shuffle=True, test_size=Parametrs["AutoEncoder_test_size"], random_state=Parametrs["AutoEncoder_Seed"])
    img_rows = x_train.shape[1]
    ##ADD noise
    if GaussianNoiseStD >= 0.0:
        x_train_noisy = x_train + np.random.normal(0,GaussianNoiseStD,x_train.shape[1])*np.ptp(X)
        x_test_noisy = x_test + np.random.normal(0,GaussianNoiseStD,x_test.shape[1])*np.ptp(X)
    if CNNModel==False :
        x_train = x_train.reshape(x_train.shape[0], img_rows).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], img_rows).astype('float32')
        x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], img_rows).astype('float32')
        x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], img_rows).astype('float32')
        X = X.reshape(X.shape[0], img_rows).astype('float32')
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], img_rows, 1).astype('float32')
        x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], img_rows, 1).astype('float32')
        x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], img_rows, 1).astype('float32')
        X = X.reshape(X.shape[0], img_rows, 1).astype('float32')
    if(Normaliztion==True):
        x_train = x_train / np.max(np.abs(X))
        x_test = x_test / np.max(np.abs(X))
        x_train_noisy = x_train_noisy/ np.max(np.abs(X))
        x_test_noisy = x_test_noisy/ np.max(np.abs(X))
        X = X / np.max(np.abs(X))
    if Normaliztion==True :
        TrainParams["Autoeoncoder_lastlayer_activation"] = "tanh"
    else:
        TrainParams["Autoeoncoder_lastlayer_activation"] = "linear"
    if CNNModel==False:
        autoencoder_model,input_img,encoded,decoded = AutoEncoder_MLP(MLPStructure, TrainParams, LayerArgList, img_rows)
    else:
        autoencoder_model,input_img,encoded,decoded = AutoEncoder_CNN(CNNStructure, TrainParams, LayerArgList, img_rows)
    initial_epoch=0
    if AutoEncoder_LoadModel==True:
        print("Loading Model")
        loaded_model = load_model(AutoEncoder_LoadModelFilename)
        ##copy layer weights
        autoencoder_model.set_weights(loaded_model.get_weights()) 
        initial_epoch=epochs
        epochs = epochs + 10
        Parametrs["AutoEncoder_TrainParams"]["epochs"] = epochs
        #print("Evaluate Model")
        #autoencoder_model.evaluate(x_eval[:,:],x_train[:,:],verbose=1)
    autoencoder_model.compile(loss=TrainParams['Loss'],
              optimizer=TrainParams['Optimizer'])
    print("Curr Model: ")
    print(autoencoder_model.summary())
    callbacks = [GetBest(monitor='val_loss', verbose=1, mode='min')]
    if GaussianNoiseStD>0.0 :
        autoencoder_model.fit(x_train_noisy, x_train,
          batch_size=TrainParams["batch_size"],
          epochs=epochs,
          validation_data= (x_test_noisy, x_test),
          callbacks = callbacks,
          initial_epoch = initial_epoch,
          verbose=1)
    else:
       autoencoder_model.fit(x_train, x_train,
          batch_size=TrainParams["batch_size"],
          epochs=epochs,
          validation_data= (x_test, x_test),
          callbacks = callbacks,
          initial_epoch = initial_epoch,
          verbose=1)
    # save model and architecture to single file
    ##autoencoder_model.save(str(AutoEncoder_SavedModelFilename)+"_ChanNum_"+str(AutoEncoder_ChanNum)+".h5")
    autoencoder_model.save(CreateBasicAutoPreTrainedFileName(Parametrs,AutoEncoder_ChanNum))
    ### https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer/55417982#55417982
    layer_name = 'encoded_vals'
    x_eval = X
    intermediate_layer_model = Model(inputs=autoencoder_model.get_input_at(0) ,
                                 outputs=autoencoder_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_eval[:,:])
    # Testing
    if CNNModel==True:
        NumofFeatures = CNNStructure[-1]
    else:
        NumofFeatures = MLPStructure[-1]
    Features = np.empty([len(x_eval), NumofFeatures])
    for idx in range(0,len(x_eval)):
        Features[idx,:] = intermediate_output[idx,:]
    if AutoEncoder_LoadModel==True:
        Parametrs["AutoEncoder_TrainParams"]["epochs"] = initial_epoch
    return(Features)


def AutoEncoderTrainFeatureExtraction(Data,Parametrs):
    RetList = []
    for CurrChan in range(0,len(Data)):
        Parametrs["AutoEncoder_ChanNum"] = CurrChan
        Parametrs["AutoEncoder_LoadModelFilename"] = CreateBasicAutoPreTrainedFileName(Parametrs,ChanNum)
        RetList.append(AutoEncoderTrainChan(Data[CurrChan][:,:],Parametrs))
    return(RetList)
