from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Variables
mes = np.linspace(1, 30, 30)
moeda1cot = np.random.uniform(3, 7, size=30)
moeda2cot = np.random.uniform(4, 8, size=30)

# train data
x = mes.reshape(-1, 1)
y1 = moeda1cot.reshape(-1, 1)
y2 = moeda2cot.reshape(-1, 1)
maxn = [max(y1), max(y2)]

# Plot the train data
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title("Cotação da moeda em R$")
ax1.set_ylabel("Cotação")
ax1.plot(x, y1, label="Moeda 1")
ax1.set_ylim(0, np.round(float(maxn[0])) + 1)
ax1.grid()
ax1.set_yticks(np.arange(0, round(float(maxn[0])) + 2, 1.0))

ax2.set_ylabel("Cotação")
ax2.set_xlabel("Dia")
ax2.plot(x, y2, label="Moeda 2", color='g')
ax2.set_ylim(0, 1 + np.round(float(maxn[1])))
ax2.grid()
ax2.set_yticks(np.arange(0, round(float(maxn[1])) + 2, 1))

# Linear Regression
regressor = LinearRegression()
regressor.fit(x, y1)
diapred = np.arange(1, 61, 1)
xpred = diapred.reshape(-1, 1)
ypred = regressor.predict(xpred)
ax1.plot(xpred, ypred, label=f"Regressão linear da moeda 1 \n $y= {float(regressor.coef_):.4f}x + {float(regressor.intercept_):.4f}$", color='y')

regressor.fit(x, y2)
ypred = regressor.predict(xpred)
ax2.plot(xpred, ypred, label=f"Regressão linear da moeda 2 \n $y= {float(regressor.coef_):.4f}x + {float(regressor.intercept_):.4f}$", color='r')

# Plot with labels
ax1.legend()
ax2.legend()
plt.show()
