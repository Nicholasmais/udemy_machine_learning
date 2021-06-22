from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data0 = pd.read_csv("startup.csv")
data = pd.get_dummies(data0)
pd.options.display.max_columns = None

x = data.drop('Profit', axis=1)
y = data['Profit']
print(x)
regressor = LinearRegression()
regressor.fit(x.iloc[:, 0:3], y)
print(regressor.coef_)

xpred = pd.DataFrame([[0, 0, 0, "0"]] * 20, columns=["R&D Spend", "Administration", "Marketing", "State"])
for j in range(0, 20):
    xpred.iloc[j, xpred.columns.get_loc("R&D Spend")] =(j+15)*regressor.coef_[0]+regressor.intercept_
    xpred.iloc[j, xpred.columns.get_loc("Administration")] =(j+15)*regressor.coef_[1]+regressor.intercept_
    xpred.iloc[j, xpred.columns.get_loc("Marketing")] =(j+15)*regressor.coef_[2]+regressor.intercept_
    xpred.iloc[j, xpred.columns.get_loc("State")] = np.random.choice(["California", "New York", "Florida"])
    '''
    xpred.iloc[j, xpred.columns.get_loc("R&D Spend")] = np.random.uniform(0.9 * x["R&D Spend"].max()/(10*(j+1)),
                                                                          1.1 * x["R&D Spend"].max()/(10*(j+1)))
    xpred.iloc[j, xpred.columns.get_loc("Administration")] = np.random.uniform(0.9 * x["Administration"].max()/(10*(j+1)),
                                                                               1.1 * x["Administration"].max()/(10*(j+1)))
    xpred.iloc[j, xpred.columns.get_loc("Marketing")] = np.random.uniform(0.9 * x["Marketing Spend"].max()/(10*(j+1)),
                                                                          1.1 * x["Marketing Spend"].max()/(10*(j+1)))
    xpred.iloc[j, xpred.columns.get_loc("State")] = np.random.choice(["California", "New York", "Florida"])
    print(np.random.uniform(0.9 * x["Marketing Spend"].max()/(1000*(j+1)),
                                                                          1.1 * x["Marketing Spend"].max()/(10*(j+1))))'''

xpred = pd.get_dummies(xpred)

ypred = regressor.predict(xpred.iloc[:, 0:3])

matrix = xpred
matrix["Predicted Profit"] = ypred

yy = [[], [], []]
yy2 = [[], [], []]

for index, row in data0.iterrows():
    if row["State"] == "California":
        yy[0].append(row["Profit"])
    elif row["State"] == "New York":
        yy[1].append(row["Profit"])
    elif row["State"] == "Florida":
        yy[2].append(row["Profit"])

for index, row in matrix.iterrows():
    if row["State_California"]:
        yy2[0].append(row["Predicted Profit"])
    elif row["State_New York"]:
        yy2[1].append(row["Predicted Profit"])
    elif row["State_Florida"]:
        yy2[2].append(row["Predicted Profit"])

plt.scatter(np.linspace(0, len(yy[0]), len(yy[0])), yy[0], label="California")
plt.scatter(np.linspace(0, len(yy[1]), len(yy[1])), yy[1], label="New York")
plt.scatter(np.linspace(0, len(yy[2]), len(yy[2])), yy[2], label="Florida")
diapred = np.arange(0,len(yy2[0]) + len(yy[0]),1)
diapred1 = np.arange(0,len(yy2[1]) + len(yy[1]),1)
diapred2 = np.arange(0,len(yy2[2]) + len(yy[2]),1)
plt.scatter(np.linspace(0 , len(yy2[0]) , len(yy2[0])), yy2[0], label="California Estimativa")
plt.scatter(np.linspace(0 , len(yy2[1]) , len(yy2[1])), yy2[1], label="New York Estimativa")
plt.scatter(np.linspace(0 , len(yy2[2]) , len(yy2[2])), yy2[2], label="Florida Estimativa")

plt.legend()
plt.grid()
plt.show()

