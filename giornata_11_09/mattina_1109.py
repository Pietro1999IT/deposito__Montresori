"""
Codice che contiene la consegna della mattina del 11/09. Caricare il dataset iris.csv,
dividere in train/test set, allenare un decision tree, stampare accuracy sul test,
visualizzare il report di sklearn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

df_path = r"C:\Users\XT286AX\OneDrive - EY\Desktop\deposito__Montresori\datasets\Iris.csv"
df = pd.read_csv(df_path)
print(f"dataset: \n\n {df.head()}")

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df[['Species']]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# WITH THE DECISION TREE
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
tree_report = classification_report(y_true=y_test, y_pred=y_pred)
print(f"the accuracy of the Decision Tree Classifier is --->{tree_accuracy}")
print(f"the classification report is \n\n {tree_report}")

# WITH THE MLP CLASSIFIER
mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train, y_train)
y_pred = mlp_classifier.predict(X_test)
mlp_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
mlp_report = classification_report(y_true=y_test, y_pred=y_pred)
print(f"the accuracy of the MLP Classifier is --->{mlp_accuracy}")
print(f"the classification report is \n\n {mlp_report}")
