# 🤖 Curso de Machine Learning - Módulo de Algoritmos Supervisionados e Não Supervisionados

Este repositório contém anotações, códigos e explicações dos principais algoritmos de Machine Learning abordados durante o curso. Abaixo você encontra a descrição teórica e prática de cada tema estudado.

---

## 📘 Índice

1. [📈 Regressão Linear](#-regressão-linear)
2. [📊 Naive Bayes](#-naive-bayes)
3. [🌳 Árvores de Decisão](#-árvores-de-decisão)
4. [🌲 Random Forest](#-random-forest)
5. [📍 KNN - K-Vizinhos Mais Próximos](#-knn---k-vizinhos-mais-próximos)
6. [📦 KMeans e Clustering](#-kmeans-e-clustering)
7. [🔗 Regras de Associação (Apriori)](#-regras-de-associação-apriori)

---

## 📈 Regressão Linear

### Introdução
A regressão linear é utilizada para modelar a relação entre uma variável dependente contínua e uma ou mais variáveis independentes.

### Conceitos:
- **Equação:** \( Y = \beta_0 + \beta_1X + \epsilon \)
- **Objetivo:** Minimizar os erros (diferença entre valores previstos e reais) usando o método dos mínimos quadrados.
- **Condições:** linearidade, normalidade dos resíduos, homocedasticidade, independência dos erros.

### Labs:
- 📎 `regressao_linear.py`: implementação com `sklearn`
- 📎 `regressao_statsmodels.ipynb`: análise estatística completa com `statsmodels`

---

## 📊 Naive Bayes

### Introdução
Classificador probabilístico baseado no Teorema de Bayes com suposição de independência entre os atributos.

### Fórmula:
\[
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
\]

### Vantagens:
- Rápido e eficaz para grandes volumes de dados
- Ideal para problemas de classificação com texto (Spam, Sentimento)

### Labs:
- 📎 `naive_bayes_texto.py`: classificação de texto com `sklearn`
- 📎 `naive_bayes_continuacao.ipynb`: continuação e avaliação de métricas

---

## 🌳 Árvores de Decisão

### Introdução
Modelo baseado em estrutura de árvore para decisão sequencial com base em atributos.

### Conceitos:
- Critérios de divisão: **Gini**, **Entropia**
- Cada nó representa uma pergunta e os ramos, possíveis respostas

### Vantagens:
- Fácil interpretação
- Pode lidar com variáveis categóricas e numéricas

### Labs:
- 📎 `arvore_decisao.py`: construção de árvore com `sklearn`
- 📎 `calculo_gini_entropia.ipynb`: (opcional) cálculo manual dos critérios de divisão

---

## 🌲 Random Forest

### Introdução
Algoritmo de *ensemble learning* baseado na combinação de múltiplas árvores de decisão.

### Conceitos:
- Cada árvore é construída com uma amostra aleatória dos dados (bootstrapping)
- A predição final é dada por votação (classificação) ou média (regressão)

### Vantagens:
- Reduz overfitting
- Alta acurácia em muitos problemas reais

### Labs:
- 📎 `random_forest.py`: treino e visualização da floresta

---

## 📍 KNN - K-Vizinhos Mais Próximos

### Introdução
Algoritmo baseado em instância que classifica um ponto com base na classe majoritária de seus vizinhos mais próximos.

### Conceitos:
- Distância Euclidiana ou outra métrica
- Valor de K influencia performance (par ou ímpar, pequeno ou grande)

### Vantagens:
- Simples e eficaz
- Não requer treinamento explícito

### Labs:
- 📎 `knn.py`: implementação manual e com `sklearn`

---

## 📦 KMeans e Clustering

### Introdução
Algoritmo de agrupamento que segmenta os dados em **K clusters** com base na distância aos centróides.

### Conceitos:
- Inicialização aleatória dos centróides
- Critério de convergência: centróides estáveis ou número de iterações

### Aplicações:
- Agrupamento de clientes, compressão de imagens, segmentação de mercados

### Labs:
- 📎 `kmeans_aglomerativo_dbscan.py`: comparação entre **KMeans**, **DBSCAN** e **Cluster Aglomerativo**
- 📎 `dendrograma.py`: dendrograma para análise hierárquica

---

## 🔗 Regras de Associação (Apriori)

### Introdução
Algoritmo para descoberta de regras de associação em grandes bases de dados transacionais (ex: carrinhos de supermercado).

### Conceitos:
- **Suporte**: frequência de um item
- **Confiança**: probabilidade condicional
- **Lift**: independência da regra

### Labs:
- 📎 `apriori.py`: regras com `mlxtend`

---

## 📂 Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend
