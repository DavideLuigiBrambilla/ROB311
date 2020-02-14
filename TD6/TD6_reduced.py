import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection, neighbors)
from numpy import shape
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score


#### Function to initialise the data
def initialize_data(file_name):
	data = pd.read_csv(file_name, header=None)
	data_label = data.iloc[:,-1]
	data = data.iloc[:,0:len(data.columns)-1]
	return data_label, data

### Initialises the train dataset
label, data = initialize_data('optdigits.tra')

### TSNE Train data-reduction
tsne = manifold.TSNE(n_components=2, init='random', random_state=None)
data_tsne = tsne.fit_transform(data)

### Function to help in the plotting for the data
cluster_helper = []
for i in range (10):
	index_helper=np.where(label==i)
	cluster_helper.append(data_tsne[index_helper])
cluster_helper = np.array(cluster_helper)


### Executes the K-means fot the reduced data dataset
centers = []
"""Initialises the k-means algorithm"""
kmeans = KMeans(n_clusters=11,random_state = 11).fit(data_tsne)
centers = kmeans.cluster_centers_
"""Get the prediction of the test dataset using the k-means boundary decision created"""
clusters = kmeans.predict(data_tsne)


### Plot raw train data
color = ['b', 'g', 'r', 'c', 'm', 'y', 'lime', 'plum', 'cyan', 'magenta']
for i in range (10):
	for j in range (shape(cluster_helper[i])[0]):
		if j==0:
			plt.scatter(cluster_helper[i][j][0],  cluster_helper[i][j][1], s=10, c=color[i], label = i, alpha=0.2)
		else:
			plt.scatter(cluster_helper[i][j][0],  cluster_helper[i][j][1], s=10, c=color[i], alpha=0.2)
for i in range (len(centers)):
	plt.scatter(centers[i][0], centers[i][1], marker = "+", s=100, c="black")
plt.legend()


##### Verification of the results, consulting the labels given in the dataset
class_list = []
print (centers[:][0], centers[:][1])
for i in range (len(centers)):
	best_dist = 1000
	for j in range (len(data_tsne)):
		dist= (((centers[i][0] - data_tsne[j][0])**2+(centers[i][1] - data_tsne[j][1] )**2)**0.5)
		if dist<best_dist:
			best_dist=dist
			index=j
	class_list.append(label[index])

pred=clusters
for i in range (len(clusters)):
	pred[i]=class_list[clusters [i]]

### Print Results
print('HOMOGENEITY SCORE:\n',metrics.homogeneity_score(label, pred))
print('F1 SCORE :\n',f1_score(label, pred, average=None))
print ('ACCURACY :\n', metrics.classification_report(label, pred))
print ('CONFUSION MATRIX :\n', confusion_matrix(label, pred))

plt.show()
