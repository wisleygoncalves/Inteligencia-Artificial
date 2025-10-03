
# 🤖 Curso de Machine Learning Avançado

Este repositório contém **anotações, códigos e explicações práticas** dos principais tópicos avançados de Machine Learning, com foco em otimização de atributos, redução de dimensionalidade, algoritmos não supervisionados avançados, técnicas para lidar com bases desbalanceadas e uso de AutoML.

---

## 📘 Índice

1. [⚙️ Engenharia e Seleção de Atributos](#️-engenharia-e-seleção-de-atributos)  
2. [📉 PCA - Principal Component Analysis](#-pca---principal-component-analysis)  
3. [🔬 Técnicas Avançadas para Clusters](#-técnicas-avançadas-para-clusters)  
4. [🏷️ Classificação Multi Label](#️-classificação-multi-label)  
5. [⚖️ Dados Desbalanceados](#️-dados-desbalanceados)  
6. [🤖 AutoML e Tunning de Modelos](#-automl-e-tunning-de-modelos)  

---

## ⚙️ Engenharia e Seleção de Atributos

### Introdução
- Processo de preparar os dados para aumentar a performance dos modelos.  
- Técnicas incluem: transformação de variáveis, criação de atributos derivados, normalização e padronização.

### Seleção de Atributos
- Métodos estatísticos (ANOVA, Qui-quadrado, Correlação de Pearson).
- Métodos baseados em modelos (Random Forest, Lasso).
- Eliminação recursiva de atributos (*Recursive Feature Elimination*).

### Labs:
- 📎 `atribute_engineering.py`  
- 📎 `feature_selection.ipynb`  

---

## 📉 PCA - Principal Component Analysis

### Introdução
- Técnica de redução de dimensionalidade.
- Converte variáveis correlacionadas em componentes ortogonais.

### Conceitos
- Maximização da variância explicada.
- Autovalores e autovetores da matriz de covariância.

### Vantagens
- Reduz ruído.
- Aumenta desempenho de modelos.
- Visualização em espaço reduzido.

### Labs:
- 📎 `pca_analysis.ipynb`  
- 📎 `lab_pca.py`

---

## 🔬 Técnicas Avançadas para Clusters

### Introdução
- Além do KMeans, outras técnicas de agrupamento podem lidar com dados complexos.

### Algoritmos
- **DBSCAN**: detecta clusters de forma arbitrária e identifica ruído.  
- **Clustering Aglomerativo**: hierárquico, gera dendrogramas.  
- **Escolha do Número Ótimo de Clusters**: métodos como Elbow, Silhouette, Hopkins e VAT.  

### Labs:
- 📎 `advanced_clustering.py`  
- 📎 `lab_clusters.ipynb`  

---

## 🏷️ Classificação Multi Label

### Introdução
- Problemas em que uma instância pode pertencer a **múltiplas classes ao mesmo tempo**.  
- Exemplo: um filme pode ser classificado como *Ação* e *Comédia*.

### Avaliação
- **Hamming Loss**  
- **Micro/Macro F1-Score**  
- **Subset Accuracy**

### Labs:
- 📎 `multi_label_classification.py`  
- 📎 `lab_multi_label.ipynb`  

---

## ⚖️ Dados Desbalanceados

### Introdução
- Problema comum em datasets reais (ex: fraudes, doenças raras).  
- Classes majoritárias dominam a predição.

### Técnicas
- **Oversampling** (SMOTE).  
- **Undersampling**.  
- **Class Weights**.  

### Labs:
- 📎 `imbalanced_data.py`  
- 📎 `lab_imbalanced.ipynb`  

---

## 🤖 AutoML e Tunning de Modelos

### Introdução
- **AutoML**: automação do processo de seleção de modelos e hiperparâmetros.  
- **Tunning**: ajuste fino com *Grid Search*, *Random Search* e *Bayesian Optimization*.  

### Ferramentas
- `sklearn.model_selection` (GridSearchCV, RandomizedSearchCV).  
- **H2O AutoML** para pipelines completos.  

### Labs:
- 📎 `automl_tunning.py`  
- 📎 `lab_automl.ipynb`  
- 📎 `lab_h2o_automl.ipynb`  

---

## 📂 Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend imbalanced-learn h2o
```

