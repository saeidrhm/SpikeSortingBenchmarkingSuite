
AutoEncoder_TrainParams = {
    "Optimizer" : 'Adam',
    "Loss" : 'MSE',
    "batch_size" : 512,
    "epochs" : 300,
    "Autoeoncoder_lastlayer_activation": 'linear'
}

AutoEncoder_LayerArg = {
    "Dense_Layer_Nodes" : 16,
    "Dense_activation" : 'relu',
    "Dropout_rate" : 0.2,
    "CNN_Num_of_Filters" : 16,
    "CNN_Filter_Size" : 5,
    "CNN_Stride" : 1,
    "CNN_MaxPooling_Size" : 2
}

def InitFeatureExtractionDeafaultParametrs():
    FeatureExtractionDeafaultParametrs = {
        "method" : 'PCA', ## 'PCA', 'tSNE' , 'DWT', 'LDAGMM', 'SSLBC','AutoEncoder'
        "PCA_OutDim" : 3,
        "tSNE_Perplexity" : 30,
        "tSNE_Step" : 250,
        "tSNE_OutDim" : 3,
        "DWT_TargetSize": 512,
        "DWT_NumofDWTKstatInputLimit": 10,
        "DWT_NumMostDistinctiveHaarDWTFeaturesCoef":0.5,
        "DWT_StrictDim":False,
        "SSLBC_SpikeLen": 128, ##128
        "SSLBC_AllLayerKernelInitializer": 'random_uniform',
        "SSLBC_LossWeghts": [1.0, 10.0, 0.0],  ##[1, 0.5, 0.5]
        "SSLBC_GNoiseSD": 0.5,##percent of PTP
        "SSLBC_Seed": 101,
        "SSLBC_Normaliztion" : True,
        "SSLBC_LoadModelFilename": 'model',
        "SSLBC_LoadModel": False,
        "SSLBC_CNNModel" : True,
        "SSLBC_MLPStructure" : [128, 64, 32, 8],
        "SSLBC_CNNStructure":[64, 32, 8],
        "SSLBC_TrainParams": AutoEncoder_TrainParams,
        "SSLBC_LayerArgList":[AutoEncoder_LayerArg for x in range(0,20)],
        "SSLBC_ClassLayerSize": 1024,
        "SSLBC_ClassLayerActivation": 'relu',
        "LDACL_sortMethodInput" : 'LDAGMM',
        "LDACL_numClusRangeInput" : 6,
        "LDACL_DimInput" : 3,
        "LDACL_maxIterInput" :30,
        "LDACL_minSampleSizeInput" : 1000,
        "LDACL_SampIterInput" : 5,
        "LDACL_minClusSizeInput" : 50,
        "LDACL_earlyStopInput" :  'false',
        "LDACL_doPlotInput" : 'false',
        "LDACL_extensionfilenameInput" : 'v15',
        "AutoEncoder_TrainParams": AutoEncoder_TrainParams,
        "AutoEncoder_MLPStructure" : [128, 64, 32, 8],
        "AutoEncoder_CNNStructure" : [64, 32, 8],
        "AutoEncoder_LayerArgList" :  [AutoEncoder_LayerArg for x in range(0,20)],
        "AutoEncoder_CNNModel" : True,
        "AutoEncoder_GaussianNoiseStD" : 0.1,
        "AutoEncoder_Normaliztion" : True,
        "AutoEncoder_FeaturesFromNoisyData" : True,
        "AutoEncoder_Seed": 101,
        "AutoEncoder_SavedModelFilename": 'model',
        "AutoEncoder_ChanNum": 0,
        "AutoEncoder_LoadModelFilename": 'model',
        "AutoEncoder_LoadModel": False,
        "AutoEncoder_BaseModelEval": [0.0, 0.0],
        "AutoEncoder_test_size": 0.25,
        "AutoEncoder_CurrCVOuter": 0,
        "AutoEncoder_CurrCVInner": 0
	}
    return(FeatureExtractionDeafaultParametrs)


def InitClusteringAndEvaluationDeafaultParametrs():
    ClusteringAndEvaluationDeafaultParametrs = {
        "ClusteringMethod" : 'kmeans', ## 'kmeans', 'GMM' , 'bayesian', 'DBSCAN', 'Spec'
        "NumofClusters" : 12,
        "Rep" : 10,
        "kmeans_InitMethod" : 'k-means++',
        "GMM_MaxIter" : 100,
        "DBSCAN_EPS" : 0.5,
        "DBSCAN_MinSamples" : 5,
        "Spectral_random_state": None,
        "Spectral_gamma" :1.0,
        "Spectral_affinity":'rbf',
        "Spectral_n_neighbors" : 10,
        "EvaluationMethods" : ["purity", "NMI"]
	}
    return(ClusteringAndEvaluationDeafaultParametrs)

