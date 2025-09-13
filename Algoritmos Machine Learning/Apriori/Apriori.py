import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os


class Apriori(object):
    base_path = r"C:\Formacao_IA\Algoritmos_Machine_Learning\Apriori"

    with open(os.path.join(base_path, 'Transacoes.txt'), 'r') as f:
        file = [line.strip().split(',') for line in f.readlines()]

    te = TransactionEncoder()
    te_ary = te.fit(file).transform(file)
    data = pd.DataFrame(te_ary, columns=te.columns_)

    def __init__(self):
        pass


    def main(self):
        print('Iniciando o Programa de Machine Learning com Apriori...')

        print('\n------Dados Analisados: ------\n')
        print(self.data)

        frequency_items = self.miner_data()

        self.rules_data(frequency_items)
    
    def miner_data(self):
        print('\n------Minerando dados: ------\n')

        frequency_items = apriori(self.data, min_support=0.5, use_colnames=True)

        print(frequency_items)

        return frequency_items


    def rules_data(self, frequency_items):
        print('\n------ Regras de Asssociação: ------\n')

        rules = association_rules(frequency_items,
                                  metric='confidence',
                                  min_threshold=0.5)
        
        print(rules)
        
        return rules


def main():
    knn = Apriori()
    knn.main()
    

if '__main__' == __name__:
    main()