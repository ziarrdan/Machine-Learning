"""
Course:         CS 7641 Assignment 3, Spring 2020
Date:           March 6th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn import random_projection
from scipy.stats import kurtosis
from scipy.linalg import pinv
import matplotlib.pyplot as plt
import scipy.sparse as sps
import numpy as np
import pandas as pd
from dataloader import dataset


# borrowed from https://github.com/JonathanTay/CS-7641-assignment-3/
def reconstructionError(projections,X):
    W = projections.getReducer().components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p @ W) @ (X.T)).T  # Unproject projected data
    errors = np.square(X - reconstructed)
    return np.nanmean(errors)


def calcFAPlotsAndReconsError(iDataset):
    retDSs = []

    for ds in iDataset:
        noOfFeaturesList = range(2, len(ds.training_x[0]), int(len(ds.training_x[0]) / 10))
        sseList = []
        for nFeatures in noOfFeaturesList:
            fa = FA(n_clusters=nFeatures)
            fa.getReducer().fit(ds.training_x)
            xTransformed = fa.getReducer().transform(ds.training_x)
            xRevTransformed = fa.getReducer().inverse_transform(xTransformed)
            sse = np.square(np.subtract(ds.training_x, xRevTransformed)).mean()
            sseList.append(sse)

        plt.style.use('seaborn-whitegrid')
        plt.plot(noOfFeaturesList, sseList, marker='o')
        plt.ylabel('Reconstruction SSE', fontsize=12)
        plt.xlabel('No. of Features', fontsize=12)
        plt.title('Reconstruction SSE Plot for ' + ds.name + ' using FA', fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/DR/Reconstruction SSE Plot for ' + ds.name + ' using FA.png')
        plt.close()

    retDS = dataset()
    fa = FA(n_clusters=40)
    fa.getReducer().fit(iDataset[0].training_x)
    xTransformed = fa.getReducer().transform(iDataset[0].training_x)
    retDS.training_x = xTransformed
    retDS.training_y = iDataset[0].training_y
    retDS.name = iDataset[0].name + ' Reduced by FA'
    retDS.build_train_test_splitSecond()
    retDSs.append(retDS)

    retDS = dataset()
    fa = FA(n_clusters=8)
    fa.getReducer().fit(iDataset[1].training_x)
    xTransformed = fa.getReducer().transform(iDataset[1].training_x)
    retDS.training_x = xTransformed
    retDS.training_y = iDataset[1].training_y
    retDS.name = iDataset[1].name + ' Reduced by FA'
    retDS.build_train_test_splitSecond()
    retDSs.append(retDS)

    return retDSs


def calcRPPlotsAndReconsError(iDataset):
    retDSs = []

    for ds in iDataset:
        noOfFeaturesList = range(2, len(ds.training_x[0]), int(len(ds.training_x[0]) / 10))
        sseList = []
        for nFeatures in noOfFeaturesList:
            sseAvg = 0
            sse = []
            for i in range(20):
                rp = RP(n_components=nFeatures, tol=0.001)
                rp.getReducer().fit(ds.training_x)
                xTransformed = rp.getReducer().transform(ds.training_x)
                reconError = reconstructionError(rp, ds.training_x)
                sseAvg += reconError
                sse.append(reconError)
            sseList.append(sseAvg / 20.)
            sseStd = np.std(sse)

        plt.style.use('seaborn-whitegrid')
        plt.plot(noOfFeaturesList, sseList, marker='o')
        plt.fill_between(noOfFeaturesList, sseList - sseStd, sseList +
                         sseStd, alpha=0.25, color='b')
        plt.ylabel('Reconstruction SSE', fontsize=12)
        plt.xlabel('No. of Features', fontsize=12)
        plt.title('Reconstruction SSE Plot for ' + ds.name + ' using RP', fontsize=12, y=1.03)
        plt.savefig('Figures/DR/Reconstruction SSE Plot for ' + ds.name + ' using RP.png')
        plt.close()

    retDS = dataset()
    rp = RP(n_components=80, tol=0.001)
    rp.getReducer().fit(iDataset[0].training_x)
    xTransformed = rp.getReducer().transform(iDataset[0].training_x)
    retDS.training_x = xTransformed
    retDS.training_y = iDataset[0].training_y
    retDS.name = iDataset[0].name + ' Reduced by RP'
    retDS.build_train_test_splitSecond()
    retDSs.append(retDS)

    retDS = dataset()
    rp = RP(n_components=9, tol=0.001)
    rp.getReducer().fit(iDataset[1].training_x)
    xTransformed = rp.getReducer().transform(iDataset[1].training_x)
    retDS.training_x = xTransformed
    retDS.training_y = iDataset[1].training_y
    retDS.name = iDataset[1].name + ' Reduced by RP'
    retDS.build_train_test_splitSecond()
    retDSs.append(retDS)

    return retDSs


# the following graphs for ICA are adapted from
# https://github.com/Heronwang/GATECH-CS7641-Machine-Learning/blob/master/Assignment3/lwang628-code.ipynb
# with changes.
def calcICAPlotsAndReconsError(iDataset):
    retDSs = []

    for ds in iDataset:
        retDS = dataset()
        ica = ICA(n_components=len(ds.training_x[0]), tol=0.001)
        ica.getReducer().fit(ds.training_x)
        xTransformedNotOrdered = ica.getReducer().transform(ds.training_x)
        order = [-abs(kurtosis(xTransformedNotOrdered[:, i])) for i in range(xTransformedNotOrdered.shape[1])]
        xTransformed = xTransformedNotOrdered[:, np.array(order).argsort()]
        ica_resNorOrdered = pd.Series([abs(kurtosis(xTransformedNotOrdered[:, i])) for i in range(xTransformedNotOrdered.shape[1])])
        ica_res = pd.Series([abs(kurtosis(xTransformed[:, i])) for i in range(xTransformed.shape[1])])
        featuresNumberCutoff = np.argmax(ica_res.values < 2.)

        plt.style.use('seaborn-whitegrid')
        ax = ica_resNorOrdered.plot(kind='bar', logy=True, label='Not Ordered Kurtosis', color='r')
        ax = ica_res.plot(kind='bar', logy=True, label='Kurtosis')
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticks(ticks[::10])
        ax.xaxis.set_ticklabels(ticklabels[::10])
        plt.axvline(featuresNumberCutoff, color='k', linestyle='--')
        plt.plot(featuresNumberCutoff, ica_res[featuresNumberCutoff], color='k', marker='o')
        plt.xlabel("Features")
        plt.ylabel("Kurtosis")
        plt.title('Components Calculated using ICA for ' + ds.name, fontsize=12, y=1.03)
        plt.savefig('Figures/DR/ICA for ' + ds.name + '.png')
        plt.close()

        ica = ICA(n_components=featuresNumberCutoff, tol=0.001)
        ica.getReducer().fit(ds.training_x)
        xRevTransformed = ica.getReducer().inverse_transform(xTransformed[:, :featuresNumberCutoff])
        sse = np.square(np.subtract(ds.training_x, xRevTransformed)).mean()
        sse = reconstructionError(ica, ds.training_x)

        print('ICA - Number of new features considering minimum of 1. for kurtosis,  ' + ds.name, ' is: ',
              featuresNumberCutoff)
        print('ICA - The reconstruction SSE considering minimum of 1. for kurtosis ' + ds.name, ' is: ', sse)

        retDS.training_x = xTransformed[:, :featuresNumberCutoff]
        retDS.training_y = ds.training_y
        retDS.name = ds.name + ' Reduced by ICA'
        retDS.build_train_test_splitSecond()
        retDSs.append(retDS)

    return retDSs


# the following graphs for PCA are adapted from
# https://github.com/Heronwang/GATECH-CS7641-Machine-Learning/blob/master/Assignment3/lwang628-code.ipynb
# with changes.
# the method for selecting the proper number of components is discussed in
# https://blogs.sas.com/content/iml/2017/08/02/retain-principal-components.html
def calcPCAPlotsAndReconsError(iDataset):
    retDSs = []

    for ds in iDataset:
        retDS = dataset()
        pca = PCAreducer(n_components=len(ds.training_x[0]))
        pca.getReducer().fit(ds.training_x)
        xTransformed = pca.getReducer().transform(ds.training_x)
        varTransformed = pd.Series(pca.getReducer().explained_variance_)
        cumVar = np.cumsum(varTransformed)
        cumVarNorm = cumVar / cumVar[len(cumVar) - 1]
        varTransformedNorm = varTransformed / cumVar[len(cumVar) - 1]
        nintyFiveVarArg = np.argmax(cumVarNorm > 0.95)

        plt.style.use('seaborn-whitegrid')
        ax = varTransformedNorm.plot(kind='bar', label='Norm. Variance')
        cumVarNorm.plot(label='Norma. Cumulative Variance')
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticks(ticks[::10])
        ax.xaxis.set_ticklabels(ticklabels[::10])
        plt.axvline(np.argmax(cumVarNorm > 0.95), color='k', linestyle='--')
        plt.plot(nintyFiveVarArg, cumVarNorm[nintyFiveVarArg], color='k', marker='o')
        plt.xlabel("Features")
        plt.ylabel("Variance")
        plt.title('Components Calculated using PCA for ' + ds.name, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/DR/PCA for ' + ds.name + '.png')
        plt.close()

        pca = PCAreducer(n_components=nintyFiveVarArg)
        pca.getReducer().fit(ds.training_x)
        xTransformed = pca.getReducer().transform(ds.training_x)
        xRevTransformed = pca.getReducer().inverse_transform(xTransformed)
        sse = np.square(np.subtract(ds.training_x, xRevTransformed)).mean()
        sse = reconstructionError(pca, ds.training_x)

        print('PCA - Number of new features considering 95% acumulative variance for ' + ds.name, ' is: ', nintyFiveVarArg)
        print('PCA - The reconstruction SSE considering 95% acumulative variance for ' + ds.name, ' is: ', sse)

        retDS.training_x = xTransformed[:, :nintyFiveVarArg]
        retDS.training_y = ds.training_y
        retDS.name = ds.name + ' Reduced by PCA'
        retDS.build_train_test_splitSecond()
        retDSs.append(retDS)

    return retDSs


class PCAreducer:
    def __init__(self, n_components=2, **kwargs):
        self.dimReducer = PCA(n_components=n_components)

    def getReducer(self):
        return self.dimReducer

    def getName(self):
        return 'PCA'


class ICA:
    def __init__(self, n_components=2, tol=0.001, **kwargs):
        self.dimReducer = FastICA(n_components=n_components, tol=tol)

    def getReducer(self):
        return self.dimReducer

    def getName(self):
        return 'ICA'


class RP:
    def __init__(self, n_components=2, **kwargs):
        self.dimReducer = random_projection.SparseRandomProjection(n_components=n_components)

    def getReducer(self):
        return self.dimReducer

    def getName(self):
        return 'RP'


class FA:
    def __init__(self, n_clusters=2, **kwargs):
        self.dimReducer = FeatureAgglomeration(n_clusters=n_clusters)

    def getReducer(self):
        return self.dimReducer

    def getName(self):
        return 'FA'
