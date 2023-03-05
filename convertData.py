import pandas as pd

datasetLearning = pd.read_csv("C:\\Users\Arthu\Downloads\datatran atualizados consolidados\datatran consolidado sem 2016.csv")
X = datasetLearning.iloc[:, :].values

dia_semana = pd.DataFrame({'dia_semana':X[:,0]})
fase_dia = pd.DataFrame({'fase_dia':X[:,1]})
condicao_metereologica = pd.DataFrame({'condicao_metereologica':X[:,2]})
tipo_pista = pd.DataFrame({'tipo_pista':X[:,3]})
tracado_via = pd.DataFrame({'tracado_via':X[:,4]})
regiao = pd.DataFrame({'regiao':X[:,5]})
causas_resumidas = pd.DataFrame({'causas_resumidas':X[:,6]})

dia_semana = pd.get_dummies(dia_semana)
fase_dia = pd.get_dummies(fase_dia)
condicao_metereologica = pd.get_dummies(condicao_metereologica)
tipo_pista = pd.get_dummies(tipo_pista)
tracado_via = pd.get_dummies(tracado_via)
regiao = pd.get_dummies(regiao)
causas_resumidas = pd.get_dummies(causas_resumidas)

new_dataset = dia_semana.join(fase_dia).join(condicao_metereologica).join(tipo_pista).join(tracado_via).join(regiao).join(causas_resumidas)
new_dataset.to_csv("new_dataset.csv")
print(new_dataset)