import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
iris = load_iris()

x=iris.data
y=iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) 

rf=RandomForestClassifier()
rf.fit(x_train,y_train)

pickle.dump(rf,open("model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))
