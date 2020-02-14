from time import time
import numpy as np
from numpy import shape
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

def separate_data(file_name):
	data = pd.read_csv(file_name)
	labels = np.array(data.iloc[:,0])
	x_data= np.array(data.iloc[:,1:(shape(data)[1])], "int16")
	DataFeatures = x_data
	return labels, DataFeatures

def convert_to_hog(x_data):
	hog_list = []
	for i in x_data:
		hf = hog(i.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(7, 7), visualize=False)
		hog_list.append(hf)
	x_to_hog = np.array(hog_list, 'float')
	normalize(x_to_hog)
	
	return x_to_hog
	
def print_results(y_test,pred):
	print ("################ ACCURACY RESULTS ################")
	print (metrics.classification_report(y_test, pred))

	print ("\n################ CONFUSION MATRIX ################")
	print (confusion_matrix(y_test,pred))

def save_mnist_image(y_test, x_mnist_test):
	for index in range (300):
		image = x_mnist_test[index][:,:]
		name = "images/" + str(y_test[:][index]) + "_" + str(index) + "_image.png"
		plt.figure()
		plt.rcParams["axes.grid"] = False
		plt.imshow(image, cmap="gray")
		plt.savefig(name)


def SVM_classifier(x_train, y_train, x_test, y_test, name_type, max_iter_value):
	initial_time = time.time()
	if max_iter_value==-2:
		classifier = svm.LinearSVC()				# Train SVM Classifier
	else:
		classifier = svm.LinearSVC()				# Train SVM Classifier
	classifier.fit(x_train, y_train.ravel())	
	pred = classifier.predict(x_test)			# Run the classifier on test data
	print_results(y_test,pred)					# Print the results of the digit recognition algorithm
	final_time = time.time()
	print ("Execution time for the ", name_type, "dataset: ", final_time-initial_time )
	return final_time-initial_time
	

#### GET DATA SET ####
y_test, x_mnist_test = separate_data('mnist_test.csv') 		#Test Data
y_train, x_mnist_train = separate_data('mnist_train.csv')	#Train Data
print("Shape of image features dataset: ", shape(x_mnist_test))

#### CONVERT DATA TO HOG FORMAT ####
x_train = convert_to_hog(x_mnist_train)	# Get the value of x_train by converting x_mnist_train to hog-normalized format
x_test = convert_to_hog(x_mnist_test)	# Get the value of x_test by converting x_mnist_test to hog-normalized format
print("Shape of hog dataset: ", shape(x_train), "\n")

#### GET IMAGE SAMPLES FROM THE DATASET ####
# ~ save_mnist_image(y_test, x_mnist_test)

#### TEST WITH THE HOG TRANSFORM
hog_time = SVM_classifier(x_train, y_train, x_test, y_test, "HOG transform",-2)

#### TEST WITH THE ORIGINAL DATA
original_time = SVM_classifier(x_mnist_train, y_train, x_mnist_test, y_test, "original",1000)

print ("Hog data set analysis is ", original_time/hog_time, " times faster")
