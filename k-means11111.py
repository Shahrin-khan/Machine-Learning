from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataset = pd.read_csv('data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

plt.scatter(x, y, label="True Position")
#plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(dataset)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()


