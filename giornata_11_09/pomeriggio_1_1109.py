"""
Codice che contiene la consegna del pomeriggio del 11/09. Caricare il dataset creditcard.csv
e allenarlo su decision tree e random forest risolvendo il problema del dataset sbilanciato 
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
    print(f"the classification report of the {model_name}is \n\n {model_report}")
    print("#####################################################################################")
    print('\n\n')

df_path = r"C:\Users\XT286AX\OneDrive - EY\Desktop\deposito__Montresori\datasets\creditcard.csv"
df = pd.read_csv(df_path)

y_series = df['Class']
y = df[['Class']]
X = df.drop('Class', axis=1)

print(f"Check to see the dataset's imbalance \n\n:{y_series.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tree = DecisionTreeClassifier(class_weight="balanced")
random_forest = RandomForestClassifier(class_weight="balanced")

apply_model(model=tree, model_name="decision tree", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
apply_model(model=tree, model_name="random forest", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Now try both models with SMOTE to balance data: \n\n")
apply_model(model=tree, model_name="decision tree", X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test, y_test=y_test)
apply_model(model=tree, model_name="random forest", X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test, y_test=y_test)


