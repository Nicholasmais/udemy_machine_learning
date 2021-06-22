import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
# Dados
# [0] custo produto
# [1] venda produto
# 3 produtos

dados = [[0, 0,0], [0, 0,0], [0, 0,0]]
lucromes = [[],[],[]]

for k in range(1,13):
    for j in range(0, 3):
        dados[j][0] = (np.random.uniform(0, 50))
        dados[j][1] = (np.random.uniform(50, 100))
        dados[j][2] = (np.random.randint(11,20))

    cliente = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for j in range(0, 3):
        cliente[j][0] = (np.random.randint(1, 15))
        cliente[j][1] = (np.random.randint(1, 15))
        cliente[j][2] = (np.random.randint(1, 15))

    lucro1 = 0
    lucro2 = 0
    lucro3 = 0
    for j in range(0,3):
            lucro1 += cliente[0][j]*dados[j][1] - dados[j][2]*dados[j][0]
            lucro2 += cliente[1][j]*dados[j][1] - dados[j][2]*dados[j][0]
            lucro3 += cliente[2][j]*dados[j][1] - dados[j][2]*dados[j][0]

    lucromes[0].append(lucro1)
    lucromes[1].append(lucro2)
    lucromes[2].append(lucro3)

mes = np.linspace(1,12,12)
mes = mes.reshape(-1,1)
lucromes[0] = np.array(lucromes[0])
lucromes[0] = lucromes[0].reshape(-1,1)
lucromes[1] = np.array(lucromes[1])
lucromes[1] = lucromes[1].reshape(-1,1)
lucromes[2] = np.array(lucromes[2])
lucromes[2] = lucromes[2].reshape(-1,1)

plt.scatter(mes,lucromes[0], label = "Lucro cliente 1")
plt.scatter(mes,lucromes[1], label = "Lucro cliente 2")
plt.scatter(mes,lucromes[2], label = "Lucro cliente 3")
plt.scatter(mes,lucromes[0]+lucromes[1]+lucromes[2], label='Lucro total')

lucromes.append(lucromes[0]+lucromes[1]+lucromes[2])

regressor = LinearRegression()
regressor.fit(mes,lucromes[0])
pred0 = regressor.predict(mes)
regressor.fit(mes,lucromes[1])
pred1 = regressor.predict(mes)
regressor.fit(mes,lucromes[2])
pred2 = regressor.predict(mes)
regressor.fit(mes,lucromes[3])
pred3 = regressor.predict(mes)

plt.plot(mes,pred0, label = "Regressão cliente 1")
plt.plot(mes,pred1, label = "Regressão cliente 2")
plt.plot(mes,pred2, label = "Regressão cliente 3")
plt.plot(mes,pred3, label = "Regressão total")

plt.grid()
plt.vlines(7,-2*np.max(lucromes),2*np.max(lucromes),color='black')
plt.hlines(0,0,13,color='black')
plt.ylim(-1.2*np.max(lucromes),1.2*np.max(lucromes))
plt.xlim(0,13)
plt.xticks(np.linspace(1,12,12))
plt.legend()
plt.show()