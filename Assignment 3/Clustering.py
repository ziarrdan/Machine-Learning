"""
Course:         CS 7641 Assignment 3, Spring 2020
Date:           March 6th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score, homogeneity_completeness_v_measure, silhouette_score
from pandas.plotting import parallel_coordinates
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import Clustering
from dataloader import dataset


def calcClusterAdded(iDataset):
    retDSs = []
    for ds in iDataset:
        if 'Income' in ds.name:
            clusterKM = 3
            clusterEM = 3

            if 'FA' in ds.name:
                clusterKM = 3
                clusterEM = 2

            if 'ICA' in ds.name:
                clusterKM = 2
                clusterEM = 2

            if 'PCA' in ds.name:
                clusterKM = 3
                clusterEM = 3

            if 'RP' in ds.name:
                clusterKM = 2
                clusterEM = 3
        elif ds.name.find('Wine'):
            clusterKM = 4
            clusterEM = 2

            if 'FA' in ds.name:
                clusterKM = 5
                clusterEM = 2

            if 'ICA' in ds.name:
                clusterKM = 3
                clusterEM = 2

            if 'PCA' in ds.name:
                clusterKM = 2
                clusterEM = 2

            if 'RP' in ds.name:
                clusterKM = 4
                clusterEM = 2

        retDS = dataset()
        emLearner = Clustering.KM(n_clusters=clusterKM)
        emLearner.getLearner().fit(ds.training_x)
        clustringY_KM = emLearner.getLearner().predict(ds.training_x)
        xTransformed = pd.concat([pd.DataFrame(ds.training_x), pd.DataFrame(clustringY_KM)], axis=1).to_numpy()
        retDS.training_x = xTransformed
        retDS.training_y = ds.training_y
        retDS.name = ds.name + ' with KM Clusters Added'
        retDS.build_train_test_splitSecond()
        retDSs.append(retDS)

        retDS = dataset()
        emLearner = Clustering.EM(n_components=clusterEM)
        emLearner.getLearner().fit(ds.training_x)
        clustringY_EM = emLearner.getLearner().predict(ds.training_x)
        xTransformed = pd.concat([pd.DataFrame(ds.training_x), pd.DataFrame(clustringY_EM)], axis=1).to_numpy()
        retDS.training_x = xTransformed
        retDS.training_y = ds.training_y
        retDS.name = ds.name + ' with EM Clusters Added'
        retDS.build_train_test_splitSecond()
        retDSs.append(retDS)

    return retDSs[0:2], retDSs[2:4]


def getClusteringEvalPlots(dataset):
    noOfClusters = range(2, 11, 1)

    for ds in dataset:
        sse = [[]]
        sil = [[[], []]]
        scores = [[[], []], [[], []], [[], []], [[], []], [[], []]]
        for cluster in noOfClusters:
            kmLearner = Clustering.KM(n_clusters=cluster)
            kmLearner.getLearner().fit(ds.training_x)
            emLearner = Clustering.EM(n_components=cluster)
            emLearner.getLearner().fit(ds.training_x)
            clustringY_KM = kmLearner.getLearner().predict(ds.training_x)
            clustringY_EM = emLearner.getLearner().predict(ds.training_x)
            homogeneityKM, completenessKM, v_measureKM = homogeneity_completeness_v_measure(ds.training_y, clustringY_KM)
            AMISKM = adjusted_mutual_info_score(ds.training_y, clustringY_KM)
            ARSKM = adjusted_rand_score(ds.training_y, clustringY_KM)
            silhouetteKM = silhouette_score(ds.training_x, clustringY_KM)
            homogeneityEM, completenessEM, v_measureEM = homogeneity_completeness_v_measure(ds.training_y, clustringY_EM)
            AMISEM = adjusted_mutual_info_score(ds.training_y, clustringY_EM)
            ARSEM = adjusted_rand_score(ds.training_y, clustringY_EM)
            silhouetteEM = silhouette_score(ds.training_x, clustringY_EM)

            sse.append(kmLearner.getLearner().inertia_)
            sil[0][0].append(silhouetteKM)
            scores[0][0].append(v_measureKM)
            scores[1][0].append(AMISKM)
            scores[2][0].append(ARSKM)
            scores[3][0].append(homogeneityKM)

            sil[0][1].append(silhouetteEM)
            scores[0][1].append(v_measureEM)
            scores[1][1].append(AMISEM)
            scores[2][1].append(ARSEM)
            scores[3][1].append(homogeneityEM)

        plt.style.use('seaborn-whitegrid')
        plt.plot(noOfClusters, sil[0][0], label='Silhouette Score, KM', marker='o')
        plt.plot(noOfClusters, sil[0][1], label='Silhouette Score, EM', marker='o', linestyle='--')
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.xlabel('K', fontsize=12)
        plt.title('Silhouette Plot for ' + ds.name, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/Clustering/Silhouette for ' + ds.name + '.png')
        plt.close()

        plt.style.use('seaborn-whitegrid')
        plt.plot(noOfClusters, scores[0][0], label='V Measure, KM', marker='o')
        plt.plot(noOfClusters, scores[1][0], label='Adj. Mutual Info, KM', marker='o')
        plt.plot(noOfClusters, scores[2][0], label='Adj. Rand. Score, KM', marker='o')
        plt.plot(noOfClusters, scores[0][1], label='V Measure, EM', marker='o', linestyle='--')
        plt.plot(noOfClusters, scores[1][1], label='Adj. Mutual Info, EM', marker='o', linestyle='--')
        plt.plot(noOfClusters, scores[2][1], label='Adj. Rand. Score, EM', marker='o', linestyle='--')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('K', fontsize=12)
        plt.title('Score Plot for ' + ds.name, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/Clustering/Score for ' + ds.name + '.png')
        plt.close()


def getTsnePlot(dataset):
    for ds in dataset:
        if 'Income' in ds.name:
            clusterKM = 3
            clusterEM = 3
            paletteKM = ['red', 'green', 'blue']
            paletteEM = ['red', 'green', 'blue']

            if 'FA' in ds.name:
                clusterKM = 3
                clusterEM = 2
                paletteKM = ['red', 'green', 'blue']
                paletteEM = ['red', 'blue']

            if 'ICA' in ds.name:
                clusterKM = 2
                clusterEM = 2
                paletteKM = ['red', 'blue']
                paletteEM = ['red', 'blue']

            if 'PCA' in ds.name:
                clusterKM = 3
                clusterEM = 3
                paletteKM = ['red', 'green', 'blue']
                paletteEM = ['red', 'green', 'blue']

            if 'RP' in ds.name:
                clusterKM = 2
                clusterEM = 3
                paletteKM = ['red', 'blue']
                paletteEM = ['red', 'green', 'blue']
        elif ds.name.find('Wine'):
            clusterKM = 4
            clusterEM = 2
            paletteKM = ['red', 'orange', 'green', 'blue']
            paletteEM = ['red', 'blue']

            if 'FA' in ds.name:
                clusterKM = 5
                clusterEM = 2
                paletteKM = ['red', 'green', 'blue', 'orange', 'purple']
                paletteEM = ['red', 'blue']

            if 'ICA' in ds.name:
                clusterKM = 3
                clusterEM = 2
                paletteKM = ['red', 'green', 'blue']
                paletteEM = ['red', 'blue']

            if 'PCA' in ds.name:
                clusterKM = 2
                clusterEM = 2
                paletteKM = ['red', 'blue']
                paletteEM = ['red', 'blue']

            if 'RP' in ds.name:
                clusterKM = 4
                clusterEM = 2
                paletteKM = ['red', 'green', 'blue', 'orange']
                paletteEM = ['red', 'blue']

        kmLearner = Clustering.KM(n_clusters=clusterKM)
        kmLearner.getLearner().fit(ds.training_x)
        emLearner = Clustering.EM(n_components=clusterEM)
        emLearner.getLearner().fit(ds.training_x)
        clustringY_KM = kmLearner.getLearner().predict(ds.training_x)
        clustringY_EM = emLearner.getLearner().predict(ds.training_x)


        homogeneityKM, completenessKM, v_measureKM = homogeneity_completeness_v_measure(ds.training_y, clustringY_KM)
        AMISKM = adjusted_mutual_info_score(ds.training_y, clustringY_KM)
        ARSKM = adjusted_rand_score(ds.training_y, clustringY_KM)
        silhouetteKM = silhouette_score(ds.training_x, clustringY_KM)
        homogeneityEM, completenessEM, v_measureEM = homogeneity_completeness_v_measure(ds.training_y, clustringY_EM)
        AMISEM = adjusted_mutual_info_score(ds.training_y, clustringY_EM)
        ARSEM = adjusted_rand_score(ds.training_y, clustringY_EM)
        silhouetteEM = silhouette_score(ds.training_x, clustringY_EM)

        print('For dataset ' + ds.name + 'using KM, the v_measure, AMIS, ARS and silhouette are: ',
              v_measureKM, AMISKM, ARSKM, silhouetteKM)
        print('For dataset ' + ds.name + 'using KM, the v_measure, AMIS, ARS and silhouette are: ',
              v_measureEM, AMISEM, ARSEM, silhouetteEM)

        if clusterKM <= 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            tsne = TSNE(n_components=2, random_state=0)
            tsne_obj = tsne.fit_transform(ds.training_x)
            ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], alpha=0.4, c=[paletteKM[x] for x in clustringY_KM])
            plt.xlabel('X')
            plt.xlabel('Y')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            tsne = TSNE(n_components=3, random_state=0)
            tsne_obj = tsne.fit_transform(ds.training_x)
            ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], tsne_obj[:, 2], alpha=0.4, c=[paletteKM[x] for x in clustringY_KM])
            plt.xlabel('X')
            plt.xlabel('Y')
            plt.xlabel('Z')

        ax.set_title('t-SNE Plot for ' + ds.name + ' using KM')
        plt.legend()
        plt.savefig('Figures/Clustering/TSNE for ' + ds.name + ' using KM.png')
        plt.close()

        if clusterEM <= 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            tsne = TSNE(n_components=2, random_state=0)
            tsne_obj = tsne.fit_transform(ds.training_x)
            ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], alpha=0.4, c=[paletteEM[x] for x in clustringY_EM])
            plt.xlabel('X')
            plt.xlabel('Y')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            tsne = TSNE(n_components=3, random_state=0)
            tsne_obj = tsne.fit_transform(ds.training_x)
            ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], tsne_obj[:, 2], alpha=0.4, c=[paletteEM[x] for x in clustringY_EM])
            plt.xlabel('X')
            plt.xlabel('Y')
            plt.xlabel('Z')

        ax.set_title('t-SNE Plot for ' + ds.name + ' using EM')
        plt.legend()
        plt.savefig('Figures/Clustering/TSNE for ' + ds.name + ' using EM.png')
        plt.close()

        if 'Wine' in ds.name:
            plt.style.use('seaborn-whitegrid')
            classDf = pd.DataFrame(clustringY_KM)
            classDf.columns = ['Class']
            datasetDf = pd.concat((pd.DataFrame(ds.training_x), classDf), axis=1)
            parallel_coordinates(datasetDf, 'Class', color=paletteKM)
            plt.xlabel('Features', fontsize=12)
            plt.title('Parallel Coordinates Plot for ' + ds.name + ' using KM', fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/Clustering/Parallel Coord for ' + ds.name + ' using KM.png')
            plt.close()

            plt.style.use('seaborn-whitegrid')
            classDf = pd.DataFrame(clustringY_EM)
            classDf.columns = ['Class']
            datasetDf = pd.concat((pd.DataFrame(ds.training_x), classDf), axis=1)
            parallel_coordinates(datasetDf, 'Class', color=paletteEM)
            plt.xlabel('Features', fontsize=12)
            plt.title('Parallel Coordinates Plot for ' + ds.name + ' using EM', fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/Clustering/Parallel Coord for ' + ds.name + ' using EM.png')
            plt.close()


class KM:
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, **kwargs):
        self.learner = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, **kwargs)

    def getLearner(self):
        return self.learner


class EM:
    def __init__(self, n_components=1, max_iter=100, n_init=1, init_params='kmeans', **kwargs):
        self.learner = GaussianMixture(n_components=n_components, max_iter=max_iter,
                                       n_init=n_init, init_params=init_params, **kwargs)

    def getLearner(self):
        return self.learner