import numpy as np
from numpy.random import seed
import os
import random
from SSDWT import *
from SSPCA import *
from SSLDAGMM import *
from SSTSNE import *
from LoadingData import *
from DeafaultParams import *
from ClusteringandEvaluation import *

MaxNumofParThreads = "3"

os.environ["OMP_NUM_THREADS"] = MaxNumofParThreads # export OMP_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = MaxNumofParThreads # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = MaxNumofParThreads # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = MaxNumofParThreads # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = MaxNumofParThreads # export NUMEXPR_NUM_THREADS

def FeatureExtraction(Data,Parametrs):
    if Parametrs['method']=='PCA':
        return PCAFeatureExtraction(Data,Parametrs)
    elif Parametrs['method']=='tSNE':
        return tSNEFeatureExtraction(Data,Parametrs)
    elif Parametrs['method']=='DWT':
        return DWTFeatureExtraction(Data,Parametrs)
    elif Parametrs['method']=='SSLBC':
        return SSLBetweenChanFeatureExtraction(Data,Parametrs)
    elif Parametrs['method']=='LDACl':
        return ClLDAFeatureExtraction(Data,Parametrs)
    elif Parametrs['method']=='AutoEncoder':
        return AutoEncoderTrainFeatureExtraction(Data,Parametrs)
    else:
        print("Cant recognize the method!")

def main():
    TemplateNamePre = "/home/saeid/spike_trains_cell_"
    TemplateNamePost = "_NoInh_6_NoExt_6_noise_25_SS_68_TS_11_RS_42_cell_based_overlapping_suppressed.npy"
    NoCells = 12
    NoChannels = 4
    CurrMainSeed = 101
    SpikeChannelDataList,SpikeChannelLabelList = LoadingSimDataOverlappingSuppressed(TemplateNamePre,TemplateNamePost,NoCells,NoChannels)
    seed(CurrMainSeed)
    os.environ['PYTHONHASHSEED']=str(CurrMainSeed)
    random.seed(CurrMainSeed)
    UltResList = list()
    for method in ['PCA','DWT']:
        FeatureExtractionDeafaultParametrs = InitFeatureExtractionDeafaultParametrs()
        FeatureExtractionDeafaultParametrs['method'] = method
        FeatureExtractionDeafaultParametrs['DWT_StrictDim'] = True
        for outdim in range(2,5):
            FeatureExtractionDeafaultParametrs['DWT_NumofDWTKstatInputLimit'] = outdim
            FeatureExtractionDeafaultParametrs['PCA_OutDim'] = outdim
            ExtractedFeatures = FeatureExtraction(SpikeChannelDataList,FeatureExtractionDeafaultParametrs)
            RetList = list()
            for clusteringmethod in ['kmeans','GMM']:
                ClusteringAndEvaluationDeafaultParametrs = InitClusteringAndEvaluationDeafaultParametrs()
                ClusteringAndEvaluationDeafaultParametrs["NumofClusters"] = NoCells
                ClusteringAndEvaluationDeafaultParametrs['ClusteringMethod'] = clusteringmethod
                for CurrChan in range(0,NoChannels):
                    CurrContext = {
                        'CurrChan' : CurrChan,
                        'method' : method,
                        'PCA_OutDim' : outdim,
                        'clusteringmethod' : clusteringmethod
                        }
                    CurrEval = ClusteringAndEvaluation(ExtractedFeatures[CurrChan],SpikeChannelLabelList[CurrChan],ClusteringAndEvaluationDeafaultParametrs)
                    RetList.append([CurrContext,CurrEval])
                    print("CurrContext: "+str(CurrContext))
                    print("CurrEval: "+str(CurrEval))
            ##save Retlist
            SaveResult(str(method)+"_"+str(outdim)+"_v1.pkl",RetList)
            UltResList.append(RetList)
    ##save or print Results
    print(UltResList)


if __name__ == "__main__":
    main()

