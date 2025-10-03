from hopkins import *
from metric import *
from visual_assessment_of_tendency import *

from sklearn import datasets
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

class Cluster(object):
    iris = datasets.load_iris()

    clust_1 = scale(iris.data)
    clust_2 = scale(np.random.rand(150, 4))


    def __init__(self):
        pass


    def process_cluster(self):
        print('APLICANDO TECNICAS AVANÇADAS DE CLUSTERS...\n')

        print(f'\nCLUSTER I:\n {self.clust_1}\n')
        print(f'\nCLUSTER II:\n {self.clust_2}\n')

        self.elbow()
        self.stats_hopkins()
        self.graph_vat()
        self.n_cluster_definite()
    

    def elbow(self):
        for cluster in [self.clust_1, self.clust_2]:
            inertia = []  

            for i in range(1, 8):
                kmeans = KMeans(n_clusters=i, n_init='auto')
                kmeans.fit(cluster)
                inertia.append(kmeans.inertia_)
            
            plt.plot(range(1, 8), inertia, marker='o')
            plt.title('Método do Cotovelo (Elbow)')
            plt.xlabel('Número de clusters (k)')
            plt.ylabel('Inércia')
            plt.show()

            inertia.clear()
    

    def stats_hopkins(self):
        print('\n[+] APLICANDO ESTATÌSTICA DE HOPKINS:')
        print(f'CLUSTERS I: {hopkins(self.clust_1, 150)}')
        print(f'CLUSTERS II: {hopkins(self.clust_2, 150)}\n')
    

    def graph_vat(self):
        print('\n[+] APLICANDO TÉCNICAS VISUAIS [VAT e iVAT]:\n')

        # VAT
        vat(self.clust_1)
        plt.title("VAT - Cluster 1")
        plt.show()

        vat(self.clust_2)
        plt.title("VAT - Cluster 2")
        plt.show()

        # iVAT
        ivat(self.clust_1)
        plt.title("iVAT - Cluster 1")
        plt.show()

        ivat(self.clust_2)
        plt.title("iVAT - Cluster 2")
        plt.show()
    

    def n_cluster_definite(self):
        print('\n[+] DEFININDO NÙMERO DE CLUSTERS - PELA MÈTRICA:\n')

        m_1_sil = assess_tendency_by_metric(self.clust_1, 'silhouette', 5)
        m_1_dav = assess_tendency_by_metric(self.clust_1, 'davies_bouldin', 5)
        m_1_cal = assess_tendency_by_metric(self.clust_1, 'calinski_harabasz', 5)

        print(m_1_sil)
        print(m_1_dav)
        print(m_1_cal)

        print('\n ---------------------------------------- \n')

        m_2_sil = assess_tendency_by_metric(self.clust_2, 'silhouette', 5)
        m_2_dav = assess_tendency_by_metric(self.clust_2, 'davies_bouldin', 5)
        m_2_cal = assess_tendency_by_metric(self.clust_2, 'calinski_harabasz', 5)

        print(m_2_sil)
        print(m_2_dav)
        print(m_2_cal)

        print('\n[+] DEFININDO NÙMERO DE CLUSTERS - PELA MÈDIA:\n')

        m_1 = assess_tendency_by_mean_metric_score(self.clust_1, 5)
        print(m_1)

        m_2 = assess_tendency_by_mean_metric_score(self.clust_2, 5)
        print(m_2)

def main():
    cl = Cluster()
    cl.process_cluster()

if __name__ == '__main__':
    main()