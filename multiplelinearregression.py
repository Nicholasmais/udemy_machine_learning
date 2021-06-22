from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv("startup.csv")
data = pd.get_dummies(data)
pd.options.display.max_columns = None

x = data.drop('Profit', axis=1)
y = data['Profit']

regressor = LinearRegression()
regressor.fit(x, y)

xpred = pd.DataFrame([[0, 0, 0, "0"]] * 20, columns=["R&D Spend", "Administration", "Marketing", "State"])
for j in range(0, 20):
    xpred.iloc[j, xpred.columns.get_loc("R&D Spend")] = np.random.uniform(0, 120000)
    xpred.iloc[j, xpred.columns.get_loc("Administration")] = np.random.uniform(90000, 120000)
    xpred.iloc[j, xpred.columns.get_loc("Marketing")] = np.random.uniform(100000, 300000)
    xpred.iloc[j, xpred.columns.get_loc("State")] = np.random.choice(["California", "New York", "Florida"])
xpred = pd.get_dummies(xpred)

ypred = regressor.predict(xpred)

matrix = xpred
matrix["Predicted Profit"] = ypred

print("Valores de treino")
print(data)

print("\nValores estimados")
print(matrix)
