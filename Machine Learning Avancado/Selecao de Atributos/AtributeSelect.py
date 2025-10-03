import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest

class AtributeSelect(object):
    path_data = r'C:\Formacao_IA\Engenharia de Atributos\Selecao de Atributos\ad.data'
    data = pd.read_csv(path_data, header=None)
    
    def __init__(self):
        pass


    def process_select_atribute(self):
        print('APLICANDO ENGENHARIA DE ATRIBUTOS [SELEÇÂO DE ATRIBUTOS]...\n')

        print('DADOS ANALISADOS:\n', self.data, '\n')

        print('SEPARANDO VARIÁVEIS X e Y...')
        
        y = self.data.iloc[:,-1].values
        X = self.data.iloc[:,:-1].values

        print('\n', f'[X]: {X}\n', '\n' f'[Y]: {y}\n')

        X_train, X_test, y_train, y_test = self.train_test_data(y, X)
        model = self.model_train(X_train, y_train)
        predict = self.model_test(X_test, model)
        self.metrics_model(y_test, predict)

        X_select = self.select_atribute_model(X, y)
        X_train_select, X_test_select, y_train_select, y_test_select = self.train_test_data(y, X_select)
        model_select = self.model_train(X_train_select, y_train_select)
        predict_select = self.model_test(X_test_select, model_select)
        self.metrics_model(y_test_select, predict_select)

    def train_test_data(self, y, X):
        print('\n Definindo conjunto de Treino e Teste...\n')

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=0)
        
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

        list_metrics = [{
            'Acurácia': accuracy,
        }]

        data_metrics = pd.DataFrame(list_metrics)

        print(data_metrics, '\n')
    

    def select_atribute_model(self, X, y):
        select = SelectKBest(chi2,
                             k=7)
        
        X_select = select.fit_transform(X, y)

        print('APLICANDO SELEÇÂO DE VARIÀVEIS:\n', X_select, '\n')

        return X_select

def main():
    ats = AtributeSelect()
    ats.process_select_atribute()


if __name__ == '__main__':
    main()