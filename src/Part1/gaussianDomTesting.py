from __future__ import division


import numpy as np; 
import random;
from random import random; 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import warnings
import math
import copy
import time
from numpy.linalg import inv,det
from gaussianMixtures import Gaussian
from gaussianMixtures import GM








'''
test1 = GM(1,1,1); 
test2 = GM(2,1,0.05); 

low = 0;
high = 10; 
num = 1000; 
x = np.linspace(low,high,num); 

aPlot = test1.plot(low=low,high = high,num=num,vis = False); 
bPlot = test2.plot(low=low,high = high,num=num,vis=False); 


plt.plot(x,aPlot); 
plt.plot(x,bPlot); 

plt.show(); 
'''



meansDif = []; 
allWeights = []; 
for i in range(0,1000):
	means = [random()*5,random()*5,random()*5,random()*5]; 

	meansDif.append([means[0]-means[2],means[0]-means[3],means[1]-means[2],means[1]-means[3]]); 
	weight = 100; 
	flag = False;
	while(flag == False):
		flag = True; 
		weight = weight - 1; 
		 
		test1 = GM(means[0],1,100);
		test1.addG(Gaussian(means[1],1,100));  
		test2 = GM(means[2],1,weight); 
		test2.addG(Gaussian(means[3],1,weight)); 

		low = 0;
		high = 5; 
		num = 1000; 
		x = np.linspace(low,high,num); 

		aPlot = test1.plot(low=low,high = high,num=num,vis = False); 
		bPlot = test2.plot(low=low,high = high,num=num,vis=False); 

		dif = [0]*len(aPlot); 

		for i in range(0,len(aPlot)):
			dif[i] = aPlot[i]-bPlot[i]; 
			if(aPlot[i] > bPlot[i]):
				tmp = True; 
			else:
				flag = False;
				break;  
				
	allWeights.append(weight); 
	


meansDifFinal = [[0 for i in range(0,len(meansDif))] for j in range(0,4)] 
count= 0; 
for i in range(0,len(meansDif)):
	for j in range(0,4):
		meansDifFinal[j][i] = meansDif[i][j]; 
		 



fig,axarr = plt.subplots(4); 
for j in range(0,4):
	axarr[j].scatter(allWeights,meansDifFinal[j]); 
 
plt.show();








