import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC


class Unbal(object):
    path_file_csv = r'C:\Formacao_IA\Engenharia de Atributos\Dados Desbalanceados\credit_simple.csv'
    data = pd.read_csv(path_file_csv, sep=';')


    def __init__(self):
        pass


    def process_desbalancean_data(self):
        print('TRABALAHANDO COM DADOS DESBALANCEADOS...\n')

        print(f'CARREGANDO OS DADOS:\n {self.data}\n')

        print(f'VERIFICANDO CLASSES DESBALANCEADA:\n {self.data.groupby(['CLASSE']).size()}\n')

        y = self.data['CLASSE'].values
        X = self.data.iloc[:,:-1].values

        print(f'\n-------- Variavel Dependente [VD]: -------- \n{y}\n')
        print(f'\n-------- Variavel Independente [VI]: -------- \n{X}\n')

        X_encoder = self.transform_variable_X(X)

        X_res, y_res = self.model_train(X_encoder, y)

        print('ANALISANDO TRANSFORMAÇÂO:\n')

        df = pd.DataFrame({'CLASSE': y_res})
        print(df.value_counts())
    

    def transform_variable_X(self, X):
        print('Transformando a Variável X [LABEL ENCONDER]...\n')

        label_encoder = LabelEncoder()

        X_encoder = X

        for i in range(X_encoder.shape[1]):
            if X_encoder[:,i].dtype == 'object':
                X_encoder[:,i] = label_encoder.fit_transform(X_encoder[:, i])
        
        print(X_encoder)
        
        return X_encoder
    

    def model_train(self, X, y):
        print('\nAplicando Modelo..\n')

        model = SMOTENC(random_state = 0,
                        categorical_features = [3, 5, 6])
        
        X_res, y_res = model.fit_resample(X, y)

        print(f'X_res:\n{X_res}\n')
        print(f'y_res: \n{y_res}\n')

        return X_res, y_res
    

def main():
    ub = Unbal()
    ub.process_desbalancean_data()


if __name__ == '__main__':
    main()