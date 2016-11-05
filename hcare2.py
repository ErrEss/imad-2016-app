from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import scipy
from pylab import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def euc(a,b):
	return distance.euclidean(a,b)
	
class MyKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
	def predict(self, x_test):
		for row in x_test:
			label = self.closest(row)
			predictions = label
		return predictions
        def calcaccuracy(self,x_test):
		caccuracy = []
		for row in x_test:
			label = self.closest(row)
			caccuracy.append(label)
		return caccuracy
		
	def closest(self, row):
		best_dist = euc(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = euc(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]		


x = np.loadtxt("train2.csv",delimiter=",")
y =np.loadtxt("label.csv",delimiter=",")

clf = MyKNN()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)

clf.fit(x_train,y_train)
cross_valid=clf.calcaccuracy(x_test)
clf.fit(x, y)
test = np.loadtxt("contacts.csv",delimiter=",")
test = test.reshape(1, -1)
predictions = clf.predict(test)

if (predictions == 0):
 f1='0'
if (predictions == 1):
 f1='1'

fd = file('result.csv','w')
fd.write(f1)
fd.close()

#print predictions
accuracy = accuracy_score(y_test, cross_valid)
accuracy = accuracy * 100
accuracy = str(accuracy)
fd1 = file('resultac.csv','w')
fd1.write(accuracy)
fd1.close()
#print cross_valid
#print accuracy


