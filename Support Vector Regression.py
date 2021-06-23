import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/SampleData.csv')
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

xdata = np.array(x)
ydata = np.array(y)


regressor = SVR(kernel='rbf')
regressor.fit(xdata.reshape(-1,1),np.ravel(y).reshape(-1,1))
ypred = regressor.predict(xdata.reshape(-1,1))

plt.scatter(xdata,ydata, label='Data', edgecolors="k")
plt.scatter(xdata,ypred, label="SVR", edgecolors="k")

plt.show()
