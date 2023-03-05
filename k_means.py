import pandas as pd
from sklearn.cluster import KMeans

# Lê o arquivo CSV em um DataFrame do pandas
data = pd.read_csv('new_dataset.csv.csv')

# Seleciona as colunas de interesse para o agrupamento
X = data[['coluna1', 'coluna2', 'coluna3']]

# Cria um modelo k-means com 4 clusters
kmeans = KMeans(n_clusters=4)

# Executa o agrupamento k-means nos dados
kmeans.fit(X)

# Imprime os centróides dos clusters
print(kmeans.cluster_centers_)

# Imprime as etiquetas de cluster para cada registro
print(kmeans.labels_)