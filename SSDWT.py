from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy

def DWTFeatureExtractionChan(Data,Parametrs):
    x = np.linspace(0, Data.shape[1], Data.shape[1])
    xvals = np.linspace(0, Data.shape[1], Parametrs['DWT_TargetSize'])
    IntepolSpikeHaar = np.zeros((Data.shape[0],Parametrs['DWT_TargetSize']))
    for idx in range(0,Data.shape[0]):
        y = Data[idx,:]
        interpfunction = interp1d(x, y, kind='cubic')
        IntepolSpike = interpfunction(xvals)
        haar4L = pywt.wavedec(IntepolSpike,'haar',level=4)
        IntepolSpikeHaar[idx,:] = np.concatenate(haar4L)
    ## kolmogrov smirnow test derived approach
    ## source: https://github.com/csn-le/wave_clus/blob/testing/Batch_files/test_ks.m
    ## source: https://github.com/csn-le/wave_clus/blob/testing/Batch_files/wave_features.m
    ksstat_arr = np.zeros(IntepolSpikeHaar.shape[1])
    for count in range(0,IntepolSpikeHaar.shape[1]) :
        thr_dist = np.std(IntepolSpikeHaar[:,count])
        thr_dist_min = np.mean(IntepolSpikeHaar[:,count]) - thr_dist
        thr_dist_max = np.mean(IntepolSpikeHaar[:,count]) + thr_dist
        Fidx = IntepolSpikeHaar[:,count]>thr_dist_min
        Sidx = IntepolSpikeHaar[:,count]<thr_dist_max
        aux = IntepolSpikeHaar[np.logical_and(Fidx,Sidx) ,count]
        x = aux
        n = x.shape[0]
        x = np.sort(x)
        ## Get cumulative sums 
        yCDF = np.linspace(1/n, 1, n)
        ## Remove duplicates; only need final one with total count
        notdup = (np.concatenate((np.diff(x), np.array([1]))) > 0)
        x_expcdf = x[notdup]
        y_expcdf = np.concatenate((np.array([0.]), yCDF[notdup]))
        ##
        ## The theoretical CDF (theocdf) is assumed to be normal  
        ## with unknown mean and sigma    
        zScores  =  (x_expcdf - np.mean(x))/np.std(x)
        #theocdf  =  normcdf(zScores , 0 , 1)
        mu = 0 
        sigma = 1 
        theocdf = 0.5 * scipy.special.erfc(-(zScores-mu)/(np.sqrt(2)*sigma)) 
        ##
        ## Compute the Maximum distance: max|S(x) - theocdf(x)|.
        ##
        end_idx = len(y_expcdf)
        delta1 = y_expcdf[range(0,(end_idx-1))] - theocdf;   ## Vertical difference at jumps approaching from the LEFT.
        delta2 = y_expcdf[range(1,end_idx)]   - theocdf;   ## Vertical difference at jumps approaching from the RIGHT.
        deltacdf = np.abs(np.concatenate((delta1 ,delta2)))    
        KSmax =  np.max(deltacdf)
        if(len(aux)>Parametrs['DWT_NumofDWTKstatInputLimit']):
            ksstat_arr[count] = KSmax
        else:
            ksstat_arr[count] = 0
    ##find most distinctive haar dwt features
    order_ksstat = (-ksstat_arr).argsort()
    order_ksstat_orig = order_ksstat
    max_ksstat = ksstat_arr[order_ksstat[0],]
    #print(max_ksstat)DWTFeatureExtraction
    #print(num_most_distinctive_haar_dwt_features_coef*max_ksstat)
    #print(ksstat_arr[order_ksstat,1])
    idx = np.array(range(0, len(ksstat_arr)))
    if(max_ksstat>0):
       high_p_val_temp = idx[(ksstat_arr[order_ksstat]<(Parametrs['DWT_NumMostDistinctiveHaarDWTFeaturesCoef']*max_ksstat))]
       upper_idx = len(order_ksstat)-1
       if(len(high_p_val_temp)>0):
           upper_idx = np.min(high_p_val_temp)
       else:
           upper_idx = len(order_ksstat)
       order_ksstat = order_ksstat[0:upper_idx]
    else:
       order_ksstat = order_ksstat[0:8]
    if(Parametrs['DWT_StrictDim']==False):
        return IntepolSpikeHaar[:,order_ksstat]
    else:
        return IntepolSpikeHaar[:,order_ksstat_orig[0:Parametrs['DWT_NumofDWTKstatInputLimit']]]


def DWTFeatureExtraction(Data,Parametrs):
    RetList = []
    for CurrChan in range(0,len(Data)):
        RetList.append(DWTFeatureExtractionChan(Data[CurrChan][:,:],Parametrs))
    return(RetList)

