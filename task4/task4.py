import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("winequality-red.csv", sep=';')


data['quality'] = (data['quality'] >= 6).astype(int)


X = data.drop('quality', axis=1)
y = data['quality']

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Parametry do przeszukania
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# Grid Search z walidacją krzyżową
svc = SVC()
grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Najlepszy model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Wyniki
print("Najlepsze parametry:", grid.best_params_)
print("Dokładność na zbiorze testowym:", accuracy_score(y_test, y_pred))
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred))
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Porównanie z regresją logistyczną
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\nModel bazowy: Regresja logistyczna")
print("Dokładność:", accuracy_score(y_test, y_pred_lr))
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred_lr))
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred_lr))