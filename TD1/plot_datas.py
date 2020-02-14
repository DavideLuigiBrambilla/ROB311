import numpy as np
import matplotlib.pyplot as plt
from TD1 import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


"""
Evaluate the influence of the k
"""
def k_variation_test(x):

    k_variation = np.zeros((1,4))
    k = 1
    kj = 0
    for ki in range (9):
        if(x == 1):
      		Id_number, x_data, y_data = load_data_winsconsin()
    	if(x == 2):
      		x_data, y_data = load_data_haberman()
              
        x_data = x_data.tolist()
        y_data = y_data.tolist()
        percentage = 0.3
        x_test, y_test, x_train, y_train = train_and_test(x_data, y_data, percentage)
        
        if ki%3 == 0:
            media = 0
            k+=2

        y_pred = k_nearest_neighbours(x_test, x_train, y_train, k)
        media += accuracy(y_pred, y_test)
        
        if ki%3 == 2:
            k_variation[0][kj] = media/3
            k_variation[0][3] += media/3
            kj+=1
            print "Confusion matrix for k=",k,"\n", conf_matrix(y_pred, y_test)
    k_variation[0][3] = k_variation[0][3]/3
    print k_variation



"""
Evaluate the influence of the test_percentage
"""  
def testSize_variation(x):

    k_variation = np.zeros((1,4))
    k = 5
    kj = 0
    percentage = 0.3
    flag = 0
    for ki in range (9):
        if(x == 1):
      		Id_number, x_data, y_data = load_data_winsconsin()
    	if(x == 2):
      		x_data, y_data = load_data_haberman()
              
        x_data = x_data.tolist()
        y_data = y_data.tolist()
        
        if ki%3 == 0:
            media=0
            if flag==1:
                percentage -=0.15
            if flag==2:
                percentage -=0.10
            flag +=1
            
        x_test, y_test, x_train, y_train = train_and_test(x_data, y_data, percentage)

        y_pred = k_nearest_neighbours(x_test, x_train, y_train, k)
        media += accuracy(y_pred, y_test)
        
        if ki%3 == 2:
            k_variation[0][kj] = media/3
            k_variation[0][3] += media/3
            kj+=1
            print "Confusion matrix for %test=",percentage, "(Test size: ", len(x_test) , ")","\n", conf_matrix(y_pred, y_test)
    k_variation[0][3] = k_variation[0][3]/3
    print k_variation



"""
Plot the 2D repartition of the data sets 
"""
def plot_data_database(x_test, y_test, y_pred, k, accuracy,groups, local, option):
    x1 = 0
    x2 = 0  
    legend_count = 0
    
    circle_size1 = 100
    circle_size2 = 50
    for u in range (len(groups)):
        for uu in range (len(groups)):
            class_label = [0,0,0,0]
            plt.figure(figsize=(6,4.5))
            plt.title('KNN method (k = %i, Accuracy = %3.4f)'%(k, accuracy))
            for i in range (len(x_test)):
                if y_pred[i] == option[0]:
                    if y_test[i]==option[1]:
                        if class_label[0]==0:
                            class_label[0]=1
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=1, c="darkgreen", edgecolors='none', s=circle_size2, label="False y = " + str(option[0]))
                        else:
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=1, c="darkgreen", edgecolors='none', s=circle_size2)
                    else:
                        if class_label[1]==0:
                            class_label[1]=1
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="lime", edgecolors='none', s=circle_size1, label="True y = " + str(option[0]))
                        else:
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="lime", edgecolors='none', s=circle_size1)                        
                    
                        
                        
                    plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="lime", edgecolors='none', s=circle_size1)
                else:
                    if y_test[i]==option[0]:
                        if class_label[2]==0:
                            class_label[2]=1
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="red", edgecolors='none', s=circle_size2, label="False y = "+ str(option[1]))
                        else:
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="red", edgecolors='none', s=circle_size2)
                    else:
                        if class_label[3]==0:
                            class_label[3]=1
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="magenta", edgecolors='none', s=circle_size1, label="True y = "+ str(option[1]))
                        else:
                            plt.scatter(x_test[i][x1], x_test[i][x2], alpha=0.7, c="magenta", edgecolors='none', s=circle_size1)

                plt.xlabel(groups[x1])
                plt.ylabel(groups[x2])
            plt.legend(loc=2)
            plt.savefig(local + '/KNN_' + str(x1) +'_'+ str(x2) + '.png')
            x2+=1
        x2=0
        x1+=1


"""
Plot the 3D repartion of the data
"""
def plot_data_haberman_3D(x_test, y_test, y_pred, k, accuracy):

    plt.figure(figsize=(12,15))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('KNN method (k = %i, Accuracy = %3.4f)'%(k, accuracy))
    
    class_label = [0,0,0,0]
    for i in range (len(x_test)):
        if y_pred[i]==1:
            if y_test[i]==2:
                if class_label[0]==0:
                    class_label[0]=1
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], alpha=0.8, c="yellow", edgecolors='none', s=40, label="False y = 1")
                else:
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], alpha=0.8, c="yellow", edgecolors='none', s=40)
            else:
                if class_label[1]==0:
                    class_label[1]=1
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], alpha=0.8, c="lime", edgecolors='none', s=60, label="True y = 1")
                else:
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], alpha=0.8, c="lime", edgecolors='none', s=60)

        else:
            if y_test[i]==1:
                if class_label[2]==0:
                    class_label[2]=1
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], marker='o', alpha=0.8, c="indigo", edgecolors='none', s=40, label="False y = 2")            
                else:
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], marker='o', alpha=0.8, c="indigo", edgecolors='none', s=40)            
            else:
                if class_label[3]==0:
                    class_label[3]=1
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], marker='o', alpha=0.8, c="magenta", edgecolors='none', s=60, label="True y = 2")            
                else:
                    ax.scatter(x_test[i][0], x_test[i][1], x_test[i][2], marker='o', alpha=0.8, c="magenta", edgecolors='none', s=60)            
    ax.set_xlabel("Age")
    ax.set_ylabel("Year of operation")
    ax.set_zlabel("Nodes detected")
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.05))#loc=2)
    plt.savefig('Imagens/haberman/KNN_haberman.png')
    


"""
Fonction that permets to call the right fonction for the right data set:
1 for winsconsin
2 for haberman
"""
def generate_images(x, x_test, y_test, y_pred, k):
    if(x == 1):
        groups = ["Clump thickness", "Uniformity of cell size", "Uniformity of cell shape", "Marginal Adhesion", "Single epithelial cell size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
        local = 'Imagens/wisconsin'
        option = [2,4]
        plot_data_database(x_test, y_test, y_pred, k, accuracy(y_pred, y_test), groups, local, option)
    if(x == 2):
        plot_data_haberman_3D(x_test, y_test, y_pred, k, accuracy(y_pred, y_test))
        local = 'Imagens/haberman'
        groups = ["Age", "Year of operation", "Nodes detected"]
        option = [1,2]
        plot_data_database(x_test, y_test, y_pred, k, accuracy(y_pred, y_test), groups, local, option)
