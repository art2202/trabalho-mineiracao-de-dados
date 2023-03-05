import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

datasetLearning = pd.read_csv("new_dataset.csv")

# axis=1 significa coluna, drop("causas_resumidas_Condutor", axis=1) = apagar coluna causas_resumidas_condutor
x_dataset = datasetLearning.drop("causas_resumidas_Condutor", axis=1)

y_dataset = datasetLearning['causas_resumidas_Condutor']

x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = train_test_split(x_dataset, y_dataset, test_size=0.2)

clf = DecisionTreeClassifier()

clf = clf.fit(x_dataset_train,y_dataset_train)

result = clf.predict(x_dataset_test)

print("Accuracy:",metrics.accuracy_score(y_dataset_test, result))