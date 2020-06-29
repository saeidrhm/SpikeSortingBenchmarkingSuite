from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

##source: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity_score(y_pred, y_true):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)).astype(float) / np.sum(contingency_matrix).astype(float) 

def ClusteringEvaluation(ClusteringLabels,Labels,MethodList):
    RetList = list()
    if 'purity' in MethodList:
        RetList.append(purity_score(ClusteringLabels.astype(float),Labels.astype(float)))
    if 'NMI' in MethodList:
        RetList.append(normalized_mutual_info_score(ClusteringLabels.astype(float),Labels.astype(float)))
    return RetList

def ClusteringAndEvaluation(Data,Labels,Parametrs):
    if Parametrs['ClusteringMethod']=='kmeans':
        kmeans = KMeans(n_clusters=Parametrs['NumofClusters'],init=Parametrs['kmeans_InitMethod'],n_init=Parametrs['Rep']).fit(Data)
        ClusteringLabels = kmeans.predict(Data)
        return ({"ClusteringLabels": ClusteringLabels, "ClusteringEvalution": ClusteringEvaluation(ClusteringLabels, Labels, Parametrs['EvaluationMethods'])})
    elif Parametrs['ClusteringMethod']=='GMM':
        gmm = GaussianMixture(n_components=Parametrs['NumofClusters'],n_init=Parametrs['Rep'],max_iter= Parametrs['GMM_MaxIter']).fit(Data)
        ClusteringLabels = gmm.predict(Data)
        return ({"ClusteringLabels": ClusteringLabels, "ClusteringEvalution": ClusteringEvaluation(ClusteringLabels, Labels, Parametrs['EvaluationMethods'])})
    elif Parametrs['ClusteringMethod']=='DBSCAN':
        DBSCANclustering = DBSCAN(eps=Parametrs['DBSCAN_EPS'],min_samples=Parametrs['DBSCAN_MinSamples']).fit(Data)
        ClusteringLabels = DBSCANclustering.labels_
        return ({"ClusteringLabels": ClusteringLabels, "ClusteringEvalution": ClusteringEvaluation(ClusteringLabels, Labels, Parametrs['EvaluationMethods'])})
    elif Parametrs['ClusteringMethod']=='Spectral':
        Specclustering = SpectralClustering(n_clusters=Parametrs['NumofClusters'], eigen_solver=None, random_state=Parametrs['Spectral_random_state'], n_init=Parametrs['Rep'], gamma=Parametrs['Spectral_gamma'], affinity=Parametrs['Spectral_affinity'], n_neighbors=Parametrs['Spectral_n_neighbors']).fit(Data)
        ClusteringLabels = Specclustering.labels_
        return ({"ClusteringLabels": ClusteringLabels, "ClusteringEvalution": ClusteringEvaluation(ClusteringLabels, Labels, Parametrs['EvaluationMethods'])})
    else:
        print("Cant recognize the method!")
