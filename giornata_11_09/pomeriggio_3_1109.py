"""
Codice che aggiunge la parte di cross validation al decision tree del precedente
script.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def apply_model(model, model_name, X_train, y_train, X_test, y_test) -> None:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    model_report = classification_report(y_true=y_test, y_pred=y_pred)
    print(f"the accuracy of the {model_name} is --->{model_accuracy}")
    print(f"the classification report of the {model_name} is \n\n {model_report}")
    print('\n\n')


df_path = r"C:\Users\XT286AX\OneDrive - EY\Desktop\deposito__Montresori\datasets\creditcard.csv"
df = pd.read_csv(df_path)

y = df[['Class']]
X = df.drop('Class', axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tree = DecisionTreeClassifier(class_weight="balanced")
auc_tree = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")

print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")
