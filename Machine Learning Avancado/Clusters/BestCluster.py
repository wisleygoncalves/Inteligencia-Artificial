import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd

class BestCluster(object):
    iris = datasets.load_iris()
    scaler = StandardScaler()
    data = scaler.fit_transform(iris.data)


    def __init__(self):
        pass


    def process_cluster(self):
        print('DEFININDO MELHOR CLUSTER...\n')

        print(f'CARREFANDO DADOS:\n {self.data}\n')

        results = self.compare_algorithms(self.data, 10)
        print(f'RESULTADOS:\n {results}\n')

        df_results = pd.DataFrame(results, columns=['Agrupador', 'Clusters', 'Score'])
        print(f'RESULTADOS PELO DATAFRAME:\n {df_results}\n')

        max_score_index = df_results['Score'].idxmax()

        print(f'OBTENDO MAIOR SCORE:\n {df_results.loc[max_score_index]}')
    

    def compare_algorithms(self, X, max_clusters):
        print('\nCOMPARNDO ALGORITMOS...\n')

        results = []
        cluster_range = range(2, max_clusters + 1)

        print('\nOBTENDO RESULTADOS DO KMeans...\n')
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=0,
                            n_init='auto')
            
            clusters = kmeans.fit_predict(X)

            silhouette_avg = silhouette_score(X, clusters)

            results.append(('KMeans', n_clusters, silhouette_avg))
        
        print('\nOBTENDO RESULTADOS DO Aglomerativo...\n')
        
        for n_clusters in cluster_range:
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            
            clusters = agglomerative.fit_predict(X)

            silhouette_avg = silhouette_score(X, clusters)

            results.append(('Agglomerative', n_clusters, silhouette_avg))
        
        print('\nOBTENDO RESULTADOS DO DBSCAN...\n')
        
        eps_values = np.arange(0.1,0.9,0.1)

        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5)

            clusters = dbscan.fit_predict(X)

            if len(set(clusters)) > 1:
                silhouette_avg = silhouette_score(X, clusters)

                results.append(('DBSCAN', eps, silhouette_avg))
        
        return results


def main():
    bc = BestCluster()
    bc.process_cluster()

if __name__ == '__main__':
    main()