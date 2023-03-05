from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

datasetLearning = pd.read_csv("new_dataset.csv")

X = datasetLearning.iloc[:, :].values

# axis=1 significa coluna, drop("causas_resumidas_Condutor", axis=1) = apagar coluna causas_resumidas_condutor
x_dataset = datasetLearning.drop("causas_resumidas_Condutor", axis=1)

y_dataset = datasetLearning['causas_resumidas_Condutor']

x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = train_test_split(x_dataset, y_dataset, test_size=0.2)


clf = svm.SVC(kernel='linear')

clf.fit(x_dataset_train, y_dataset_train)

result = clf.predict(x_dataset_test)

accuray = accuracy_score(result, y_dataset_test)
f1 = f1_score(result, y_dataset_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)