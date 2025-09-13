from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os


class KNN(object):
    base_path = r"C:\Formacao_IA\Algoritmos_Machine_Learning\KNN"

    data = pd.read_csv(os.path.join(base_path, 'mt_cars.csv'))
    data.drop(data.columns[0], axis=1, inplace=True)


    def __init__(self):
        pass


    def main(self):
        print('Iniciando o Programa de Machine Learning com KNN...')

        print('\n------Dados Analisados: ------\n')
        print(self.data)

        print('\n------ Separando as Variaveis (VD[y] VS VI[x]) ------\n')

        y = self.data['cyl'].values
        X = self.data[['mpg', 'hp']].values

        print(f'\n-------- Variavel Dependente [VD]: -------- \n{y}\n')
        print(f'\n-------- Variavel Independente [VI]: -------- \n{X}\n')

        X_train, X_test, y_train, y_test = self.train_test_data(y, X)

        model = self.knn_model(X_train, y_train)

        predict = self.model_test(X_test, model)

        self.metrics_model(y_test, predict)
    

    def train_test_data(self, y, X_encoder):
        print('\n Definindo conjunto de Treino e Teste...\n')

        X_train, X_test, y_train, y_test = train_test_split(X_encoder,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=12)
        
        print(f'\n[+] X_Train:\n{X_train}\n')
        print('-----------------------')

        print(f'\n[+] X_test:\n{X_test}\n')
        print('-----------------------')

        print(f'\n[+] y_train:\n{y_train}\n')
        print('-----------------------')

        print(f'\n[+] y_test:\n{y_test}\n')
        print('-----------------------')
        
        return X_train, X_test, y_train, y_test
    
    def knn_model(self, X, y):
        print('\nTreinando Modelo..\n')

        knn = KNeighborsClassifier(n_neighbors=3)
        model = knn.fit(X, y)

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
        precision = precision_score(y_test, predict, average='weighted')
        recall = recall_score(y_test, predict, average='weighted')
        f1 = f1_score(y_test, predict, average='weighted')

        list_metrics = [{
            'Acurácia': accuracy,
            'Precisão': precision,
            'Recall': recall,
            'F1': f1
        }]

        data_metrics = pd.DataFrame(list_metrics)

        print(data_metrics)

        print('\n-------------- [+] Matriz de Confusão.. --------------\n')

        cm = confusion_matrix(y_test, predict)
        print(cm)
    

def main():
    knn = KNN()
    knn.main()
    

if '__main__' == __name__:
    main()