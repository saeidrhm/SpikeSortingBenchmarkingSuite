from sklearn.manifold import TSNE

def tSNEFeatureExtractionSingleChan(Data,Parametrs):
    X_embedded = TSNE(n_components=Parametrs['tSNE_OutDim'],n_iter=Parametrs['tSNE_Step'], perplexity=Parametrs['tSNE_Perplexity']).fit_transform(Data)
    return(X_embedded)

def tSNEFeatureExtraction(Data,Parametrs):
    RetList = []
    for CurrChan in range(0,len(Data)):
        RetList.append(tSNEFeatureExtractionSingleChan(Data[CurrChan][:,:],Parametrs))
    return(RetList)

