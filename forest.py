from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from random import uniform

iris = datasets.load_iris()

data = pd.DataFrame({
    'Sepal Length':iris.data[:,0],
    'Sepal Width' : iris.data[:,1],
    'Petal Length' : iris.data[:,2],
    'Petal Width' : iris.data[:,3],
    'Species' : iris.target
    
    })

X = data[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]]
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

values = [uniform(0,1)*max(iris.data[:,i]) for i in range (0,4)]
pred = [[values[0],values[1],values[2],values[3]]]
nome = clf.predict(pred)

print(f"{iris.feature_names[0]} = {pred[0][0]}, {iris.feature_names[1]} = {pred[0][1]}, {iris.feature_names[2]} = {pred[0][2]},"
    f" {iris.feature_names[3]} = {pred[0][3]} "
    f"\nType = {iris.target_names[nome]}")
