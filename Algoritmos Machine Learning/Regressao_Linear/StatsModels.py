import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns
import os

class LinearRegression(object):
    base_path = r"C:\Formacao_IA\Algoritmos_Machine_Learning\Regressao_Linear"

    data = pd.read_csv(os.path.join(base_path, 'mt_cars.csv'))
    data.drop(data.columns[0], axis=1, inplace=True)


    def __init__(self):
        pass


    def main(self):
        print('Iniciando o Programa de Regressão Linear com StatsModels...')

        print('\n------Dados Analisados: ------\n')
        print(self.data)

        self.corr()
        self.graphic_dispersion()

        print('\n------ Testando Modelo I: ------\n')
        
        model_1 = self.model_rlm('mpg ~ wt + disp + hp')
        self.residual_model(model_1)
        self.test_normal_data(model_1)

        print('\n------ Testando Modelo II: ------\n')
        
        model_2 = self.model_rlm('mpg ~ disp + cyl')
        self.residual_model(model_2)
        self.test_normal_data(model_2)

        print('\n------ Testando Modelo III: ------\n')
        
        model_3 = self.model_rlm('mpg ~ drat + vs')
        self.residual_model(model_3)
        self.test_normal_data(model_3)
    

    def corr(self):
        print('\nGerando o Correlograma')

        corr = self.data.corr()

        plt.figure(figsize=(10, 8)) 
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=' .2f')

        plt.show()
    

    def graphic_dispersion(self):
        print('\nGerando o Gráfico de Dispersão')

        columns = [c for c in self.data.columns if c != 'mpg']
        n_plots = len(columns)

        ncols = 3
        nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
        axes = axes.flatten()

        for i, name_column in enumerate(columns):
            sns.scatterplot(x='mpg', y=name_column, data=self.data, ax=axes[i])

            axes[i].set_ylabel(name_column, fontsize=10)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.0) 
        plt.show()
    

    def model_rlm(self, formula):
        print('\nGerando Modelo de Regressão Linear Múltipla...\n')

        model = sm.ols(formula=formula, data=self.data)
        model = model.fit()
        
        print(model.summary())

        return model


    def residual_model(self, model):
        print('\nGerando Gráfico dos Resíduos Padronizados...')

        residual = model.resid

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histograma
        axes[0].hist(residual, bins=20, color='gray', edgecolor='white')
        axes[0].set_xlabel('Resíduos')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Histograma de Resíduos')

         # Q-Q Plot
        stats.probplot(residual, dist='norm', plot=axes[1])
        axes[1].set_title('Q-Q Plot de Resíduos')

        plt.tight_layout()
        plt.show()
    

    def test_normal_data(self, model):
        print('\nTeste de Normalidade dos Dados...\n')

        stat, pval = stats.shapiro(model.resid)

        print(f'------ Shapiro-Wilk ----- \n[+] Estatística: {stat:.3f} \n[+] p-value: {pval:.3f} ')

        if pval <= 0.05:
            print('\n[CONCLUSÃO]: Rejeita-se H0 ao nível de significância de 5%,' \
            ' assim os resíduos não são normalmente distribuídos\n')
        
        if pval > 0.05:
            print('\n[CONCLUSÃO]: Aceita-se H0 ao nível de significância de 5%,' \
            ' assim os resíduos são normalmente distribuídos\n')


def main():
    lg = LinearRegression()

    lg.main()
    

if '__main__' == __name__:
    main()