import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt

dataset = np.array(
    [['Asset Flip', 100, 1000],
     ['Text Based', 500, 3000],
     ['Visual Novel', 1500, 5000],
     ['2D Pixel Art', 3500, 8000],
     ['2D Vector Art', 5000, 6500],
     ['Strategy', 6000, 7000],
     ['First Person Shooter', 8000, 15000],
     ['Simulator', 9500, 20000],
     ['Racing', 12000, 21000],
     ['RPG', 14000, 25000],
     ['Sandbox', 15500, 27000],
     ['Open-World', 16500, 30000],
     ['MMOFPS', 25000, 52000],
     ['MMORPG', 30000, 80000]
     ])

x = dataset[:, 1].astype(int)
y = dataset[:,2].astype(int)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

regressor = DecisionTreeRegressor(random_state=0)
Y = regressor.fit(x,y)

plt.scatter(x,y, label='Dataset')
plt.plot(x,Y.predict(x),label="Tree regression")
plt.legend()
plt.show()

export_graphviz(regressor, out_file='tree.dot',feature_names=['Production Cost'])
#http://www.webgraphviz.com/ 