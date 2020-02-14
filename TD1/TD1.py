import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import pandas as pd
from math import sqrt
from sklearn.metrics import confusion_matrix
from plot_datas import *
import os

"""
string2int()
This function is used to convert the 16 missing attribute values in the
database to a "0" value, in order to handle better those exceptions
"""
def string2int(sti):
    try:
        return int(sti)
    except ValueError:
        return 0


"""
def load_data_winsconsin()
Reading the winsconsin dataset given in TD: we elminate the 16 samples that 
contains less characteristics: the ones with ?
"""
def load_data_winsconsin():
    data = pd.read_csv('breast-cancer-wisconsin.data', header=None)
    data_columns = int(shape(data)[1])
    data_lignes = int(shape(data)[0])-16
    data_vector = np.zeros((data_lignes,data_columns)).astype(int)
    
    jj=0
    flag=0
    for j in range (data_lignes):
        flag=0
        if data[6][j] == "?":
            flag = 1
        for i in range (data_columns):
            if flag == 1:
                jj = j+1        
                data_vector[j][i] = data[i][jj]
            else:
                data_vector[j][i] = string2int(data[i][j])

    Id_number = data_vector[:,0]
    x_data = data_vector[:,1:-1]
    y_data = data_vector[:,-1]
    
    return Id_number, x_data, y_data



"""
def load_data_haberman()
We read the haberman dataset given in TD
"""
def load_data_haberman():
    data = pd.read_csv('haberman.data', header=None)
    
    data_columns = int(shape(data)[1])
    data_lignes = int(shape(data)[0])
    data_vector = np.zeros((data_lignes,data_columns)).astype(int)
    
    for i in range (data_columns):
       for j in range (data_lignes):
           data_vector[j][i] = string2int(data[i][j])
    
    x_data = data_vector[:,0:-1]
    y_data = data_vector[:,-1]
    
    return x_data, y_data



"""
def train_and_test(x, y, test_percent)
In order to create our training set and our test set we are going to
receive a percentage that we will use in order to split the datas 
between train and set. A small percentage will create a more reliable
dataset. 
"""
def train_and_test(x, y, test_percent):
    sample_test = int(len(x) * test_percent)
    x_test = []
    y_test = []
    x_train = []
    y_train = []    
    for i in range (sample_test):
        j = random.randint(0,len(x)-1)
        x_test.append(x[j])
        y_test.append(y[j])
        x.pop(j)
        y.pop(j)
    x_train = x
    y_train = y
    return x_test, y_test, x_train, y_train



"""
def calculate_distance(x_test, x_train)
In order to calculate the distance we have noticed that the vector
is composed by 9 characteristics so we are going to calculate the
distance on all these characteristics. We are going to use differents distances:
1: Euclidean, 2: Manhattan and 3: Chebyshev. We will obtain the distance from
one vector from an other vector in a multi-dimension environnement.
"""
def calculate_distance(x_test, x_train, method):
    d = 0
    d_new = 0
    for i in range(len(x_test)):
        
        """
        1: Euclidean
        2: Manhattan
        3: Chebyshev 
        """
        if method == 1:
            d = d + (x_train[i] - x_test[i])**2
        if method == 2:
            d = d + abs(x_train[i] - x_test[i])
        if method == 3:
            d_new = abs(x_train[i] - x_test[i])
            if d_new > d:
                d = d_new
    if method == 1:
        return sqrt(d)
    else:
        return d



"""
def count_most_common (y_voisin)
Vote Majoritaire fonction
Calculation between the k-nearest-neighbours the most frequent classe
of data in order to classify the sample that we are treating. 
"""
def count_most_common (y_voisin):
    y_voisin.sort() 
    number_classes = list(set(y_voisin))
    majoritaire = [0 , 0]
    for i in range (len(number_classes)):
        u = y_voisin.count(number_classes[i])
        if i == 0:
            majoritaire = [number_classes[i], u]
        if u > majoritaire [1]:
            majoritaire = [number_classes[i], u]
    return majoritaire[0]



"""
def k_nearest_neighbours 
Implementation of KNN  algorithme: we are going to give to the fonction the test and train
values in order to test that the algorithme works well. It is necessary to choose a
value for k to pass it as parameters. The value that we are giving back is the predicted class.
We can choose which type of distance we will use by specifying the method: 1 for Euclidean, 2
for Manhattan and 3 for Chebyshev
"""
def k_nearest_neighbours(x_test, x_train, y_train, k):
    y_pred = []
    for test in x_test:
        i = 0
        y = []
        distances = []
        for train in x_train:
            #Options(3rd argument of calculate_distance): 1: Euclidean, 2: Manhattan , 3: Chebyshev 
            distance = calculate_distance(test, train, 3)
            distances.append([distance,i])
            i = i + 1
        min_dist = sorted (distances, key=lambda x: x[0])

        for j in range(k):
            y.append(y_train[min_dist[0][1]])
        y_pred.append(count_most_common(y))
    return y_pred



"""
conf_matrix(y_pred, y_test)
We are going to calculate the confusion matrix in order to see for each classes
which sample is well classified and which sample is wrong classified
"""
def conf_matrix(y_pred, y_test):
    classes = list(set(y_pred))
    classes_size = len(classes)
    #print classes_size
    confusion = np.zeros((classes_size, classes_size))
    #print shape(confusion)

    for i in range (classes_size):
        for j in range (classes_size):
            for k in range (len(y_pred)):
                if y_pred[k] == classes[i]:
                    if i==j:
                        if y_pred[k]==y_test[k]:
                            confusion[i][j]+=1
                    else:
                        if y_pred[k]!=y_test[k]:
                            confusion[j][i]+=1
    return confusion


"""
def accuracy(y_pred,y_test)
We are going to calculate the accuracy of the implemented algoirthm
It's the ratio between the correct previsions and the total of the data
"""
def accuracy(y_pred,y_test):
    num = 0.0
    for i in range(len(y_pred)):
        if(y_pred[i] == y_test[i]):
            num = num + 1
    return num/(len(y_pred))


"""
Function in order to create the folder in which we are going to save
all the plots fo the test data
"""
def create_image_folder():
    import shutil
    folder1 = 'Imagens'
    folder2 = 'Imagens/haberman'
    folder3 = 'Imagens/wisconsin'
    
    try:
        shutil.rmtree(folder1)
    except OSError:
        print ("Deletion of the directory failed")
    else:
        print ("Successfully deleted the directory")                                      
    
    try:
        # Create target Directory
        os.mkdir(folder1)
        os.mkdir(folder2)
        os.mkdir(folder3)
        print("Directory created ") 
    except:
        print("Directory already exists")
    
"""
main
"""
if __name__ == '__main__':
    while(1):	
  	x = input("Insert:\n 1 for the Breast Cancer Wisconsin Dataset.\n 2 for the Haberman's Survival Dataset\n")
	if(x == 1):
  		Id_number, x_data, y_data = load_data_winsconsin()
		break
	if(x == 2):
  		x_data, y_data = load_data_haberman()
		break
	else:
  		print("\nThe value inserted is different than 1 or 2! Try Again!\n")

    create_image_folder()
    
    x_data = x_data.tolist()
    y_data = y_data.tolist()

    percentage = 0.3
    x_test, y_test, x_train, y_train = train_and_test(x_data, y_data, percentage)
    
    k = 5
    y_pred = k_nearest_neighbours(x_test, x_train, y_train, k)

    print "\nAccuracy is", accuracy(y_pred, y_test),"\n"
    print "OURS:Confusion Matrix is\n", conf_matrix(y_pred, y_test)
    
    print "SKLEARN:Confusion Matrix is\n", confusion_matrix(y_test, y_pred),"\n"


    """
    Tests to verify the performance with different parameters: to uncomment if desired
    They will give back the confusion matrices, the accuracy and the mean of the accuracy
    """
    #k_variation_test(x)
    #testSize_variation(x)

    """
    Generate the plots
    """
    generate_images(x, x_test, y_test, y_pred, k)