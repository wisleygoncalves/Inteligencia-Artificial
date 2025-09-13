import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import os

class NaiveBayes(object):
    base_path = r"C:\Formacao_IA\Algoritmos_Machine_Learning\Naive_Bayes"
    
    data = pd.read_csv(os.path.join(base_path, 'insurance.csv'), keep_default_na=False)
    data.drop(data.columns[0], axis=1, inplace=True)

    def __init__(self):
        pass

    def main(self):
        print('Iniciando o Programa de Machine de Learning com Naive Bayes...')

        print('\n------ Dados Analisados: ------\n')
        print(self.data)

        print('\n------ Dados Nulos na Base Dados ------\n')
        print(self.data.isnull().sum())

        print('\n------ Separando as Variaveis (VD[y] VS VI[x]) ------\n')
        
        y = self.data.iloc[:,7].values
        X = self.data.drop(self.data.columns[7], axis=1).values

        print(f'\n-------- Variavel Dependente [VD]: -------- \n{y}\n')
        print(f'\n-------- Variavel Independente [VI]: -------- \n{X}\n')

        X_encoder = self.transform_variable_X(X)

        X_train, X_test, y_train, y_test = self.train_test_data(y, X_encoder)

        model = self.model_train(X_train, y_train)

        predict = self.model_test(X_test, model)

        self.metrics_model(y_test, predict)

        self.report_model(y_test, predict)

    
    def transform_variable_X(self, X):
        print('Transformando a Variável X [LABEL ENCONDER]...\n')

        label_encoder = LabelEncoder()

        X_encoder = X

        for i in range(X_encoder.shape[1]):
            if X_encoder[:,i].dtype == 'object':
                X_encoder[:,i] = label_encoder.fit_transform(X_encoder[:, i])
        
        print(X_encoder)
        
        return X_encoder
    
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

    def model_train(self, X_train, y_train):
        print('\nTreinando Modelo..\n')

        model = GaussianNB()
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
    
    def report_model(self, y_test, predict):
        print('\n-------------- [+] Relatório do Modelo.. --------------\n')

        report = classification_report(y_test, predict)
        print(report)


def main():
    nb = NaiveBayes()

    nb.main()
    

if '__main__' == __name__:
    main()
