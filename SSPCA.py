from sklearn.decomposition import PCA

def PCAFeatureExtractionSingleChan(Data,Parametrs):
    pca = PCA(n_components=Parametrs['PCA_OutDim'])
    pca.fit(Data)
    return(pca.transform(Data))

def PCAFeatureExtraction(Data,Parametrs):
    RetList = []
    for CurrChan in range(0,len(Data)):
        RetList.append(PCAFeatureExtractionSingleChan(Data[CurrChan][:,:],Parametrs))
    return(RetList)
