import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import StandardScaler

class MultiLabel(object):
    path_file_csv = r'C:\Formacao_IA\Engenharia de Atributos\Multi Label\Musica.csv'
    data = pd.read_csv(path_file_csv, sep=',')
    
    data_classe = data.iloc[:,0:6].values
    data_predict = data.iloc[:,7:78].values

    scaler = StandardScaler()
    data_predict_transform = scaler.fit_transform(data_predict)

    def __init__(self):
        pass


    def procees_multilabel(self):
        print('APLICANDO MULTI LABEL...\n')

        print(f'CARREGANDO OS DADOS:\n {self.data}\n')
        print(f'CARREGANDO AS CLASSES DOS DADOS:\n {self.data_classe}\n')
        print(f'CARREGANDO OS DADOS PARA PREVISÃO:\n {self.data_predict}\n')
        print(f'CARREGANDO OS DADOS TRANSFORMADOS PARA PREVISÃO:\n {self.data_predict_transform}\n')

        X_train, X_test, y_train, y_test = self.train_test_data(self.data_predict_transform, self.data_classe)

        model = self.model_train(X_train, y_train)

        predict = self.model_test(X_test, model)

        self.metrics_model(y_test, predict)


    def train_test_data(self, predict, classe):
        print('\n Definindo conjunto de Treino e Teste...\n')

        X_train, X_test, y_train, y_test = train_test_split(predict,
                                                            classe,
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

        model = SVC()
        classifir_chain = ClassifierChain(base_estimator=model, random_state=5)

        _model = classifir_chain.fit(X_train, y_train)

        print(_model)

        return _model


    def model_test(self, X_test, model):
        print('\nTestando Modelo..\n')

        predict = model.predict(X_test)
        print(predict)

        return predict
    

    def metrics_model(self, y_test, predict):
        print('\n-------------- [+] Métricas do Modelo.. --------------\n')

        loss = hamming_loss(y_test, predict)
        print(loss)


def main():
    ml = MultiLabel()
    ml.procees_multilabel()


if __name__ == '__main__':
    main()
