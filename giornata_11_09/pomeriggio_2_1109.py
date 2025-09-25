"""
Codice che aggiunge la parte di validation nello split al codice di pomeriggio_1109 
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val =  train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"the size of the train is {y_train.shape[0]}, the test is {y_test.shape[0]}, the validation is {y_val.shape[0]}")
tree = DecisionTreeClassifier(class_weight="balanced")

apply_model(model=tree, model_name="decision tree", X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val)
