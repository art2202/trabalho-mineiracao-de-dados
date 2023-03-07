import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score,f1_score,recall_score,precision_score)


datasetLearning = pd.read_csv("new_dataset.csv")


# axis=1 significa coluna, drop("causas_resumidas_Condutor", axis=1) = apagar coluna causas_resumidas_condutor
x_dataset = datasetLearning.drop("causas_resumidas_Condutor", axis=1)

y_dataset = datasetLearning['causas_resumidas_Condutor']


x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = train_test_split(x_dataset, y_dataset, test_size=0.2)

model = GaussianNB()

model.fit(x_dataset_train, y_dataset_train)

result = model.predict(x_dataset_test)

accuray = accuracy_score(result, y_dataset_test)

f1 = f1_score(result, y_dataset_test, average="weighted")
recall = recall_score(result,y_dataset_test,average="weighted",zero_division=0)
precision = precision_score(result, y_dataset_test,average="weighted")

print("Acurácia do Naives:", accuray)
print("F1 do Naives:", f1)
print("Cobertura do Naives", recall)
print("Precisão do Naives", precision)