import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class AutoML(object):
    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    df_X = pd.DataFrame(X, columns=cancer.feature_names)

    def __init__(self):
        pass


    def process_automl(self):
        print('TRABALAHANDO COM AUTOML...\n')

        print(f'CARREGANDO OS DADOS:\n {self.cancer}\n')
        print(f'CARREGANDO A VARIAVEL X DOS DADOS:\n {self.X}\n')
        print(f'CARREGANDO A VARIAVEL y DOS DADOS:\n {self.y}\n')
        print(f'CARREGANDO A VARIAVEL X DOS DADOS - DATAFRAME:\n {self.df_X}\n')

        X_train, X_test, y_train, y_test = self.train_test_data(self.y, self.X)

        X_train_encoder, X_test_encoder = self.transform_variable_X(X_train, X_test)

        result_models, models = self.models_crossvalidation(X_train_encoder, y_train)

        models_best = self.best_models(result_models,
                                       models, 
                                       X_train_encoder,
                                       y_train,
                                       X_test_encoder,
                                       y_test)
    

    def train_test_data(self, y, X_encoder):
        print('\n Definindo conjunto de Treino e Teste...\n')

        X_train, X_test, y_train, y_test = train_test_split(X_encoder,
                                                            y,
                                                            test_size=0.2,
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


    def transform_variable_X(self, X_train, X_test):
        print('Transformando a Variável X [LABEL ENCONDER]...\n')

        scaler = StandardScaler()

        X_train_encoder = scaler.fit_transform(X_train)
        X_test_encoder = scaler.fit_transform(X_test)

        print(f'\n[+] X_train_encoder:\n{X_train_encoder}\n')
        print('-----------------------')

        print(f'\n[+] X_test_encoder:\n{X_test_encoder}\n')
        print('-----------------------')
        
        return X_train_encoder, X_test_encoder
    

    def models_crossvalidation(self, X_train_encoder, y_train):
        print('\nANALISANDO MODELOS:\n')

        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVC': SVC(),
            'KNeighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Ada Boost': AdaBoostClassifier()
        }

        results = {}

        for name, model in models.items():
            scores = cross_val_score(model, X_train_encoder, y_train, cv=5)
            results[name] = scores.mean()
        
        for name, score in results.items():
            print(f'{name}: {score:.4f}')

        return results, models
    

    def best_models(self,
                    result_models,
                    models, 
                    X_train_encoder,
                    y_train,
                    X_test_encoder,
                    y_test):
        print('\nANALISANDO MELHORES MODELOS:\n')

        best_model_name = max(result_models, key=result_models.get)
        best_model = models[best_model_name]

        model = best_model.fit(X_train_encoder, y_train)

        y_pred = best_model.predict(X_test_encoder)

        accuracy_model = accuracy_score(y_test, y_pred)

        print(f'\nMELHOR MODELO:\n{best_model}\n')

        print(f'\nTREINANDO MELHOR MODELO:\n{model}\n')

        print(f'\nPREVISÂO DO MELHOR MODELO:\n{y_pred}\n')

        print(f'\nACURÁCIA DO MELHOR MODELO:\n{accuracy_model}\n')

        return model, y_pred

def main():
    ml = AutoML()
    ml.process_automl()


if __name__ == '__main__':
    main()