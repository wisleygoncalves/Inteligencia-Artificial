from numpy import *

class LinearRegression(object):
    x = []
    y = []
    value = 0


    def __init__(self):
        pass


    def main(self):
        print('Iniciando o Programa de Regressão Linear Manual...')

        corr = self.correlation_coef()
        b0, b1 = self.params_regression_linear(corr)
        predic_value = self.predict_value(b0, b1, self.value)

        print('\n--------- RESULTADOS --------')
        print(f'\nCorrelação: {corr}')
        print(f'\nParâmentros: [B0: {b0}] e [B1: {b1}]')
        print(f'\nPrevisão Resultado: {predic_value}')
    

    def correlation_coef(self):
        print('\nCalculando o coeficiente de Correlação...')

        cov_xy = cov(self.x, self.y, bias=True)[0, 1]
        var_x = var(self.x)
        var_y = var(self.y)

        corr = cov_xy / sqrt(var_x * var_y)

        return corr


    def params_regression_linear(self, cor):
        print('\nCalculando Parâmetros B0 e B1')

        std_x = std(self.x)
        std_y = std(self.y)

        mean_x = mean(self.x)
        mean_y = mean(self.y)

        b1 = cor * (std_y / std_x)
        
        b0 = mean_y - (b1 * mean_x )

        return b0, b1


    def predict_value(self, b0, b1, value):
        print('\nPrevendo Resultado')

        predict = b0 + (b1 *value)

        return predict


def main():
    lg = LinearRegression()

    lg.x = [1, 2, 3, 4, 5]
    lg.y = [2, 4, 6, 8, 10]
    lg.value = 6

    lg.main()
    
    
if '__main__' == __name__:
    main()