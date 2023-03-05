# importando as bibliotecas necessárias
import pandas as pd
from apyori import apriori

# carregando os dados do arquivo CSV em um DataFrame
dados = pd.read_csv('dataset_textual.csv')

# convertendo os dados do DataFrame em uma lista de listas
dados = dados.values.tolist()

# executando a análise apriori com os parâmetros mínimos de suporte e confiança
regras = apriori(dados, min_support=0.5, min_confidence=0.5)

# exibindo as regras encontradas
for regra in regras:
    print(regra)