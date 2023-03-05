import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

datasetLearning = pd.read_csv("new_dataset.csv")
print(datasetLearning.columns.size)
X = datasetLearning.iloc[:, :].values

# axis=1 significa coluna, drop("causas_resumidas_Condutor", axis=1) = apagar coluna causas_resumidas_condutor
x_dataset = datasetLearning.drop("causas_resumidas_Condutor", axis=1).drop("causas_resumidas_Outras Causas", axis=1).drop("causas_resumidas_Pista", axis=1).drop("causas_resumidas_Veiculo", axis=1)

y_dataset = pd.DataFrame({'causas_resumidas_Condutor':X[:,48]}).join(pd.DataFrame({'causas_resumidas_Outras Causas':X[:,49]}))\
    .join(pd.DataFrame({'causas_resumidas_Pista':X[:,50]})).join(pd.DataFrame({'causas_resumidas_Veiculo':X[:,51]}))

x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = train_test_split(x_dataset, y_dataset, test_size=0.2)

scaler = StandardScaler()

train = scaler.fit_transform(x_dataset_train)

test = scaler.transform(x_dataset_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_dataset_train, y_dataset_train)
result = knn.predict(x_dataset_test)

accuracy = accuracy_score(y_dataset_test, result)
print("Accuracy:", accuracy)

