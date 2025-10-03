from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

class ModelPCA(object):
    iris = datasets.load_iris()
    data_iris = iris.data
    class_iris = iris.target

    def __init__(self):
        pass

    def process_pca(self):
        print('APLICANDO ENGENHARIA DE ATRIBUTOS [PCA]...\n')

        print('DADOS ANALISADOS:\n', self.data_iris, '\n')

        print('CLASSE DOS DADOS ANALISADOS:\n', self.class_iris, '\n')

        data_scale = self.transform_scale_data()

        X_train, X_test, y_train, y_test = self.train_test_data(data_scale)
        model = self.model_train(X_train, y_train)
        predict = self.model_test(X_test, model)
        self.metrics_model(y_test, predict)

        data_pca = self.pca_model(data_scale)
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = self.train_test_data(data_pca)
        model_pca = self.model_train(X_train_pca, y_train_pca)
        predict_pca = self.model_test(X_test_pca, model_pca)
        self.metrics_model(y_test, predict_pca)
    

    def transform_scale_data(self):
        sc = StandardScaler()
        data_scale = sc.fit_transform(self.data_iris)

        print('TRANSFORMANDO A SCALA DOS DADOS:\n', data_scale, '\n')

        return data_scale
    

    def train_test_data(self, data_scale):
        print('\n Definindo conjunto de Treino e Teste...\n')

        X_train, X_test, y_train, y_test = train_test_split(data_scale,
                                                            self.class_iris,
                                                            test_size=0.3,
                                                            random_state=123)
        
        print(f'\n[+] X_Train:\n{X_train}\n')
        print('-----------------------')

        print(f'\n[+] X_test:\n{X_test}\n')
        print('-----------------------')

        print(f'\n[+] y_train:\n{y_train}\n')
        print('-----------------------')

        print(f'\n[+] y_test:\n{y_test}\n')
        print('-----------------------')
        
        return X_train, X_test, y_train, y_test
    

    def model_train(self, X_train, y_train):
        print('\nTreinando Modelo..\n')

        model = RandomForestClassifier(random_state=1234,
                                       n_estimators=100)
        
        model = model.fit(X_train, y_train)

        print(model)

        return model
    

    def model_test(self, X_test, model):
        print('\nTestando Modelo..\n')

        predict = model.predict(X_test)
        print(predict)

        return predict
    

    def metrics_model(self, y_test, predict):
        print('\n-------------- [+] Métricas do Modelo.. --------------\n')

        accuracy = accuracy_score(y_test, predict)

        list_metrics = [{
            'Acurácia': accuracy,
        }]

        print(list_metrics, '\n')
    

    def pca_model(self, data_scale):
        pca = PCA(n_components=3)

        data_pca = pca.fit_transform(data_scale)

        print('APLICANDO PCA:\n', data_pca, '\n')

        return data_pca

def main():
    mpca = ModelPCA()

    mpca.process_pca()

if __name__ == '__main__':
    main()