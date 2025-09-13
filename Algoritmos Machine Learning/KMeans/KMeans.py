from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8" 
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

class KMeansModel(object):
    _data = datasets.load_iris()
    data = datasets.load_iris().data

    def __init__(self):
        pass

    def main(self):
        print('Iniciando o Programa de Machine Learning com KMeans....')

        print('\n------ Dados Analisados: ------\n')
        print(self.data)

        model_kmeans = self.kmeans_model()
        model_dbscan = self.dbscan_model()
        model_cluster_aglomerative = self.cluster_aglomerative_model()

        titles = ['KMeans', 'DBSCAN', 'Cluster Aglomerative']
        models = [model_kmeans.labels_,
                  model_dbscan.labels_,
                  model_cluster_aglomerative.labels_]

        for i, label in enumerate(models):
            self.graph_cluster(label, titles[i])
        
        self.graph_dendogram()
    

    def kmeans_model(self):
        print('\n-----------------------------------------------------\n')
        print('\nTreinando Modelo Kmeans..\n')

        kmeans = KMeans(n_clusters=3, n_init='auto')
        model = kmeans.fit(self.data)

        print('\n-------------------- [+] Label ----------------------\n')
        
        print(model.labels_) 

        print('\n-------------- [+] Matriz de Confusão.. --------------\n')

        cm = confusion_matrix(self._data.target, model.labels_)
        print(cm)
    
        return model
    

    def dbscan_model(self):
        print('\n-----------------------------------------------------\n')
        print('\nTreinando Modelo DBSCAN..\n')

        dbscan = DBSCAN(eps=0.5,
                        min_samples=3)
        model = dbscan.fit(self.data)

        print('\n-------------------- [+] Label ----------------------\n')
        
        print(model.labels_)

        print('\n-------------- [+] Matriz de Confusão.. --------------\n')

        cm = confusion_matrix(self._data.target, model.labels_)
        print(cm)

        print('\n[ATENÇÃO]: Não faz sentindo usar matriz de confusão, devido aos ruídos do DBSCAN\n')

        return model
    

    def cluster_aglomerative_model(self):
        print('\n-----------------------------------------------------\n')
        print('\nTreinando Modelo Aglomerative Cluster..\n')

        aglomerative = AgglomerativeClustering(n_clusters=3)
        model = aglomerative.fit(self.data)

        print('\n-------------------- [+] Label ----------------------\n')
        
        print(model.labels_)

        print('\n-------------- [+] Matriz de Confusão.. --------------\n')

        cm = confusion_matrix(self._data.target, model.labels_)
        print(cm)

        return model
    

    def graph_cluster(self, labels, title):
        colors = ['black', 'red', 'green', 'purple'] 

        plt.figure(figsize=(8, 4))

        for i, c, l in zip(range(-1, 3), colors, ['Noise', 'Setosa', 'Versicolor', 'Virginica']):
            if i == -1:
                plt.scatter(self.data[labels == i, 0],
                            self.data[labels == i, 3],
                            c=c,
                            label=l,       
                            alpha=0.5,
                            s=50,
                            marker='x')
            else:
                plt.scatter(self.data[labels == i, 0],
                            self.data[labels == i, 3],
                            c=c,
                            label=l,       
                            alpha=0.5,
                            s=50)

        plt.legend()
        plt.title(title)
        plt.xlabel('Comprimento Sépala')
        plt.ylabel('Largura da Pétala')
        plt.show()

    
    def graph_dendogram(self):
        plt.figure(figsize=(12,6))
        plt.title('Cluster Hierárquico: Dendograma')
        plt.xlabel('Índice')
        plt.ylabel('Distância')

        linkage_matrix = linkage(self.data, method='ward')
        dendrogram(linkage_matrix, truncate_mode='lastp', p=15)

        plt.axhline(y=7, c='gray', lw=1, linestyle='dashed')
        plt.show()

def main():
    km = KMeansModel()
    km.main()
    

if __name__ == '__main__':
    main()
