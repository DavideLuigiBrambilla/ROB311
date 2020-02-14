import numpy as np
from numpy import shape
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

### Initialises the data
def initialize_data(file_name):
	data = pd.read_csv(file_name, header=None)
	data_label = data.iloc[:,-1]
	data = data.iloc[:,0:len(data.columns)-1]
	return data_label, data
label, data = initialize_data('optdigits.tra')
label_test, data_test = initialize_data('optdigits.tes')


## Inittialize K-means
index=[]
k_init = np.zeros((10,shape(data)[1])).astype(int)

### Creates a initialization of the k-means
for i in range (10):
	index.append(np.where(label==i)[0][0])

a = np.array(data)
k_init[:][:] = a[index][:]
k_init_values = np.array (k_init)

#Changing the value of clusters 10,11,12
kmeans = KMeans(n_clusters=10, random_state = 0).fit(data)
centers = kmeans.cluster_centers_
clusters = kmeans.predict(data_test)


class_list = []
data_test = np.array(data_test)
index = 0
print (shape(centers))
for i in range (shape(centers)[0]):

	best_dist = 9999999
	for j in range (shape(data_test)[0]):
		dist=0

		for k in range(shape(centers)[1]):
			dist = dist+((centers[i][k]-data_test[j][k])**2)
		dist = (dist)**0.5


		if dist<best_dist:
			best_dist=dist
			index=j
	class_list.append(label_test[index])

pred=clusters
for i in range (len(clusters)):
	pred[i]=class_list[clusters [i]]

print('Homogeneity score :\n', metrics.homogeneity_score(label_test, pred))
print('F1 score :\n', f1_score(label_test, pred, average=None))
print ('ACCURACY :\n', metrics.classification_report(label_test, pred))
print('Confusion matrix:\n', confusion_matrix(label_test, pred))
