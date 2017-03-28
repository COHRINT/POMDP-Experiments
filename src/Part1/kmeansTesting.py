'''
Test using k means to cluster gaussians prior to condensation using runnals
'''
from __future__ import division
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import numpy as np
from scipy.stats import multivariate_normal as mvn
import copy
from random import random
import time
import matplotlib.pyplot as plt



#Initialize some random gaussians to try merging
testGM = GM();  

numInit = 1000; 
numFinal = 50; 

for i in range(0,numInit//2):
	tmpMean = [random()*2+3,random()*2+3]; 
	offCov = random(); 
	tmpVar = [[random()+1,offCov],[offCov,random()+1]]; 
	weight = random()*20; 
	testGM.addG(Gaussian(tmpMean,tmpVar,weight)); 

for i in range(0,numInit//2):
	tmpMean = [random()*2,random()*2]; 
	offCov = random(); 
	tmpVar = [[random()+1,offCov],[offCov,random()+1]]; 
	weight = random()*20; 
	testGM.addG(Gaussian(tmpMean,tmpVar,weight));

testGM.normalizeWeights(); 


testGM2 = copy.deepcopy(testGM); 



[x1,y1,c1] = testGM.plot2D(vis=False); 



firstCondenseTime = time.clock(); 
testGM.condense(numFinal); 
testGM.normalizeWeights(); 
firstCondenseTime = time.clock() - firstCondenseTime; 

print("The time to condense without k-means: " + str(firstCondenseTime) + " seconds"); 

[x2,y2,c2] = testGM.plot2D(vis=False); 



SSE1 = 0; 
for i in range(0,len(c1)):
	for j in range(0,len(c1[i])):
		SSE1 += (c2[i][j]-c1[i][j])**2; 

print("The sum squared error without k-means: " + str(SSE1)); 



secondCondenseTime = time.clock(); 
testGM2 = testGM2.kmeansCondensationN(numFinal);
testGM2.normalizeWeights();  
secondCondenseTime = time.clock() - secondCondenseTime; 


print("Time to condense with k-means: " + str(secondCondenseTime) + " seconds"); 

[x3,y3,c3] = testGM2.plot2D(vis = False); 




SSE2 = 0; 
for i in range(0,len(c1)):
	for j in range(0,len(c1[i])):
		SSE2 += (c3[i][j]-c1[i][j])**2; 

print("The sum squared error with k-means: " + str(SSE2)); 

print(""); 

print("Error Ratio of k-means/runnals = " + str(SSE2/SSE1));

print("Time Ratio of k-means/runnals = " + str(secondCondenseTime/firstCondenseTime)); 

if(testGM.size > numFinal):
	print('Error: testGM size is: '+ str(testGM.size)); 
	testGM1.display(); 
if(testGM2.size > numFinal):
	print('Error: testGM2 size is: ' + str(testGM2.size)); 
	testGM2.display(); 


fig = plt.figure()
ax1 = fig.add_subplot(231)
con1 = ax1.contourf(x1,y1,c1, cmap=plt.get_cmap('viridis'));
ax1.set_title('Original Mixture'); 
plt.colorbar(con1); 

minNum = 100000; 
maxNum = -100000;
for i in range(0,len(c1)):
	for j in range(0,len(c1[i])):
		if(c1[i][j]<minNum):
			minNum = c1[i][j]; 
		if(c1[i][j] > maxNum):
			maxNum = c1[i][j]; 

ax2 = fig.add_subplot(232)
con2 = ax2.contourf(x2,y2,c2, cmap=plt.get_cmap('viridis'));
ax2.set_title('Condensation with Runnals only: ' + str(firstCondenseTime) + ' seconds');
plt.colorbar(con2,boundaries = np.linspace(minNum,0.0001,maxNum)); 


ax3 = fig.add_subplot(233)
con3 = ax3.contourf(x3,y3,c3, cmap=plt.get_cmap('viridis'));
ax3.set_title('Condensation with kmeans+Runnals: ' + str(secondCondenseTime) + ' seconds');
plt.colorbar(con3,boundaries = np.linspace(minNum,0.0001,maxNum)); 

ax5 = fig.add_subplot(235)
con5 = ax5.contourf(x2,y2,c2-c1, cmap=plt.get_cmap('viridis'));
ax5.set_title('Error with Runnals only');
plt.colorbar(con5); 


ax6 = fig.add_subplot(236)
con6 = ax6.contourf(x3,y3,c3-c1, cmap=plt.get_cmap('viridis'));
ax6.set_title('Error with kmeans+Runnals');
plt.colorbar(con6); 


fig.suptitle("Condensation comparison with #Initial = " + str(numInit) + " and #Final = " +str(numFinal)); 

plt.show(); 


