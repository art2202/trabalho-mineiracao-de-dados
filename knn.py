import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

datasetLearning = pd.read_csv("new_dataset.csv")

# axis=1 significa coluna, drop("causas_resumidas_Condutor", axis=1) = apagar coluna causas_resumidas_condutor
x_dataset = datasetLearning.drop("causas_resumidas_Condutor", axis=1)

y_dataset = datasetLearning['causas_resumidas_Condutor']

x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = train_test_split(x_dataset, y_dataset, test_size=0.2)

scaler = StandardScaler()

train = scaler.fit_transform(x_dataset_train)

test = scaler.transform(x_dataset_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_dataset_train, y_dataset_train)
result = knn.predict(x_dataset_test)

accuracy = accuracy_score(y_dataset_test, result)
f1 = f1_score(result, y_dataset_test, average="weighted")
recall = recall_score(result,y_dataset_test,average="weighted",zero_division=0)
precision = precision_score(result, y_dataset_test,average="weighted")

print("Acurácia do knn:", accuracy)
print("F1 do Knn:", f1)
print("Cobertura do knn", recall)
print("Precisão do knn", precision)




