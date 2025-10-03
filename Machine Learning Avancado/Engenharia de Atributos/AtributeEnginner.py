import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class AtributeEnginner(object):
    path = r"C:\Formacao_IA\Algoritmos_Machine_Learning_Avancado\Engenharia de Atributos\credit_simple.csv"

    data = pd.read_csv(path, sep=';')

    def __init__(self):
        pass

    def atribute_enginner(self):
        print('Iniciando programa de Engenharia de Atributo [TRANSFORMAÇÂO DE DADOS]...')

        print('\n', 'DADOS ANALISADOS', '\n', self.data, '\n')

        print('SEPARANDO VARIÁVEIS X e Y...')
        
        y = self.data['CLASSE']
        X = self.data.iloc[:,:-1]

        print('\n', f'[X]: {X}\n', f'[Y]: {y}\n')

        X_transform_1 = self.fix_atribute(X)
        X_transform_2 = self.fix_outliers(X_transform_1)

        X_transform_3 = self.transform_cluster(X_transform_2)
        X_transform_4 = self.transform_date(X_transform_3)
        z, X_transform_5 = self.label_enconders(X_transform_4)
        m = self.standard_scaler(X_transform_5)

        print('\nAJUSTANDO DATAFRAME\n')
        X = pd.concat([
                        X_transform_5,
                        z,
                        pd.DataFrame(m, columns=['SALDO_ATUAL_N', 'RESIDENCEADESDE_N', 'IDADE_N'])
                    ], axis=1)
        
        print(X)

        print('\nDELETANDO COLUNAS\n')

        X.drop(columns=['SALDO_ATUAL', 'RESIDENCIADESDE', 'IDADE',
                        'OUTROSPLANOSPGTO', 'DATA', 'OUTROS_banco'], inplace=True)
        
        print(X)


    def fix_atribute(self, X):
        print('\n', f'COLUNAS NULAS EM [X]: {X}\n', f'{X.isnull().sum()}\n')

        median = X['SALDO_ATUAL'].median()
        print(f'APLIANDO ENGENHARIA DE ATRIBUTO NA COLUNA SALDO ATUAL - MEDIANA...\n', median)

        X['SALDO_ATUAL'].fillna(median, inplace=True)

        print('\n', f'COLUNAS NULAS EM [X] ATUALIZADA:\n', f'{X.isnull().sum()}\n')

        moda = X.groupby(['ESTADOCIVIL']).size()
        print(f'APLIANDO ENGENHARIA DE ATRIBUTO NA COLUNA SALDO ATUAL - MODA...\n', moda)

        X['ESTADOCIVIL'].fillna('masculino solteiro', inplace=True)

        print('\n', f'COLUNAS NULAS EM [X] ATUALIZADA:\n', f'{X.isnull().sum()}\n')

        return X

    def fix_outliers(self, X):
        print('ANALISANDO OUTLIERS...\n')

        median = X['SALDO_ATUAL'].median()

        std = X['SALDO_ATUAL'].std()
        print(f'DESVIO PADRÃO: {std}, \n')

        print(X.loc[X['SALDO_ATUAL'] >= 2*std, 'SALDO_ATUAL'])

        print('\nCORRIGINDO OUTLIERS...\n')
        X.loc[X['SALDO_ATUAL'] >= 2*std, 'SALDO_ATUAL'] = median

        print('ANALISANDO OUTLIERS...\n')
        print(X.loc[X['SALDO_ATUAL'] >= 2*std, 'SALDO_ATUAL'])

        return X


    def transform_cluster(self, X):
        print('\nAPLICANDO DATA BIND NA COLUNA PROPOSITO\n')
        clusters = X.groupby('PROPOSITO').size()
        print(clusters)
        
        print('\nMELHORANDO REPRESENTIVIDADE DOS ITENS DA COLUNA PROPOSITO...\n')
        X.loc[X['PROPOSITO'] == 'Eletrodomésticos', 'PROPOSITO'] = 'outros'
        X.loc[X['PROPOSITO'] == 'qualificação', 'PROPOSITO'] = 'outros'

        print('\nANALIZANDO REPRESENTIVIDADE DOS ITENS DA COLUNA PROPOSITO\n')
        clusters = X.groupby('PROPOSITO').size()
        print(clusters)

        return X
    

    def transform_date(self, X):
        print('\nAPLICANDO DATA BIND NA COLUNA DATA\n')

        print('\nTRANSFROMADO A COLUNA PELO PANDAS...\n')
        X['DATA'] = pd.to_datetime(X['DATA'], format='%d/%m/%Y')
        print(X['DATA'])

        print('SELECIONANDO ANO, MÊS E ANO...\n')

        X['ANO'] = X['DATA'].dt.year
        X['MES'] = X['DATA'].dt.month
        X['DIA'] = X['DATA'].dt.day_name()

        print(X.iloc[:,7:10])

        return X
    

    def label_enconders(self, X):
        print('\nTRANSFORMANDO COLUNA ESTADOCIVIL(VARIAVEL CATEGORICA) em NÚMEROS\n')
        
        print('VARIÀVEIS:\n',
              '\nESTADO CIVIL:', X['ESTADOCIVIL'].unique(), '\n',
              '\nPROPOSITO:', X['PROPOSITO'].unique(), '\n',
              '\nDIA:', X['DIA'].unique(), '\n')
        
        print('APLICANDO LABEL ENCONDER\n')

        label_enconder_1 = LabelEncoder()

        X['ESTADOCIVIL'] = label_enconder_1.fit_transform(X['ESTADOCIVIL'])
        X['PROPOSITO'] = label_enconder_1.fit_transform(X['PROPOSITO'])
        X['DIA'] = label_enconder_1.fit_transform(X['DIA'])

        print(X)

        print('\nAPLICANDO VARIÁVEL DUMMY PARA COLUNA [OUTROSPLANOSPGTO] \n')

        othres = X['OUTROSPLANOSPGTO'].unique()
        print(othres)

        z = pd.get_dummies(X['OUTROSPLANOSPGTO'], prefix='OUTROS')
        print('\n', z, '\n')

        return z, X

    def standard_scaler(self, X):
        print('\nAJUSTANDO VALORES DA COLUNA\n')

        sc = StandardScaler()
        m = sc.fit_transform(X.iloc[:, 0:3])
        print(m)

        return m

def main():
    ag = AtributeEnginner()
    
    ag.atribute_enginner()

if __name__ == '__main__':
    main()