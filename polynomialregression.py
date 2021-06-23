from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1.5, 1, 100)
x = np.array(5 * t ** 4 + 4 * t ** 3 - 3 * t ** 2 + 2 * t - 1)
for i in range(0, 100):
    x[i] = x[i] * np.random.randint(7, 13) / 10

xpoly = PolynomialFeatures(1).fit_transform(x.reshape(-1, 1))

regressor = LinearRegression()
regressor.fit(xpoly, x)
ypred = regressor.predict(xpoly)

plt.scatter(t, x)
plt.plot(t, ypred, label="Polynomial Regression Sklearn PolynomialFeatures", color='r')
polyfunction = np.poly1d(np.polyfit(t, x, 5))
plt.plot(t, polyfunction(t), label='Numpy polyfit', color='g')
plt.legend()
plt.show()
