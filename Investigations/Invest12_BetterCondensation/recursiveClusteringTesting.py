
from __future__ import division
from sys import path

from gaussianMixturesTesting import Gaussian
from gaussianMixturesTesting import GM
import numpy as np
from scipy.stats import multivariate_normal as mvn
import copy
from random import random
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv,det


def hierarchicalClustering(mixture):
	allG = mixture.Gs; 

	#Upper triangular
	distMatrix = (np.ones(shape=(len(allG),len(allG)))*10000).tolist(); 


	minNum = 10000; 
	minIndex = [0,0]; 
	for i in range(0,len(allG)-1):
		for j in range(i+1,len(allG)):
			distMatrix[i][j] = distance(allG[i].mean,allG[j].mean); 
			if(distMatrix[i][j] < minNum):
				minIndex = [i,j]; 
				minNum = distMatrix[i][j]; 


	pass;


def distance(a,b):
	suma = 0; 
	for i in range(0,len(a)):
		suma+=(a[i]+b[i])**2; 
	res = np.sqrt(suma); 
	return res; 


def simpleKmeansToOne(mixture):
	means = mixture.getMeans(); 
	var = mixture.getVars(); 
	weights = mixture.getWeights(); 

	finalMean = np.zeros(shape=(1,len(means[0]))); 
	for i in range(0,len(means)):
		finalMean += np.array(means[i])*weights[i]; 

	finalVar = np.zeros(shape=(len(means[0]),len(means[0]))); 
	for i in range(0,len(means)):
		finalVar += weights[i]*np.array(var[i]); 
		finalVar += weights[i]*((np.matrix(means[i]) - np.matrix(finalMean))*(np.matrix(means[i]) - np.matrix(finalMean)).T)

	newGM = GM(finalMean.tolist()[0],finalVar.tolist(),1); 
	#newGM.plot2D(); 

	return newGM; 

if __name__ == "__main__":




	
	#Initialize some random gaussians to try merging
	# testGM = GM();  

	# numInit = 100; 
	# numFinal = 10; 

	# for i in range(0,numInit//2):
	# 	tmpMean = [random()*2+3,random()*2+3]; 
	# 	offCov = random()*0.02; 
	# 	tmpVar = [[offCov+random()*0.05,offCov],[offCov,offCov+random()*0.05]]; 
	# 	weight = random(); 
	# 	testGM.addG(Gaussian(tmpMean,tmpVar,weight)); 

	# for i in range(0,numInit//2):
	# 	tmpMean = [random()*2,random()*2]; 
	# 	offCov = random()*0.02; 
	# 	tmpVar = [[(offCov+random()*0.02),offCov],[offCov,(offCov+random()*0.02)]]; 
	# 	weight = random(); 
	# 	testGM.addG(Gaussian(tmpMean,tmpVar,weight));


	# testGM.normalizeWeights();  

	testGM = GM(); 
	testGM.addG(Gaussian([1,1.1],[[1,0],[0,1]],1)); 
	testGM.addG(Gaussian([5,4.9],[[1,0],[0,1]],1)); 
	testGM.addG(Gaussian([5,5],[[1,0],[0,1]],1)); 
	testGM.addG(Gaussian([1,1],[[1,0],[0,1]],1)); 
	
	testGM.normalizeWeights(); 
	# testGM.plot2D(); 
	#testGM.display(); 

	testGM2 = testGM.kmeansCondensationN(k=2); 
	testGM2.display(); 
	testGM3 = simpleKmeansToOne(testGM); 
	testGM2.normalizeWeights(); 
	testGM2.plot2D(); 
	testGM3.plot2D(); 

	# testGM2 = copy.deepcopy(testGM); 
	# testGMOrig = copy.deepcopy(testGM); 
	# testSimpleCondense = copy.deepcopy(testGM); 

	# [x1,y1,c1] = testGM.plot2D(vis=False); 



	# firstCondenseTime = time.clock(); 
	# testGM.condense(numFinal); 
	# testGM.normalizeWeights(); 
	# firstCondenseTime = time.clock() - firstCondenseTime; 

	# print("The time to condense without k-means: " + str(firstCondenseTime) + " seconds"); 

	# firstCondenseTimestr = str(firstCondenseTime)[0:5];

	# [x2,y2,c2] = testGM.plot2D(vis=False); 



	# isd1 = testGMOrig.ISD(testGM); 

	# print("The ISD without k-means: " + str(isd1)); 

	# #isd1str = str(isd1)[0:3]+'e'+str(isd1)[len(str(isd1))-3:]
	
	

	# secondCondenseTime = time.clock(); 
	# testGM2 = testGM2.kmeansCondensationN(k=5,kDown=2);
	# testGM2.normalizeWeights();  
	# secondCondenseTime = time.clock() - secondCondenseTime; 
	# secondCondenseTimestr = str(secondCondenseTime)[0:5];

	# print("Time to condense with k-means: " + str(secondCondenseTime) + " seconds"); 

	# [x3,y3,c3] = testGM2.plot2D(vis = False); 




	# isd2 = testGMOrig.ISD(testGM2); 

	# print("The ISD with k-means: " + str(isd2)); 
	# #isd2str = str(isd2)[0:3]+'e'+str(isd2)[len(str(isd2))-3:]
	

	# #print(""); 

	# #print("Time Ratio of k-means/runnals = " + str(secondCondenseTime/firstCondenseTime)); 

	# #print("Error Ratio of k-means/runnals = " + str(isd2/isd1)); 

	# if(testGM.size > numFinal):
	# 	print('Error: testGM size is: '+ str(testGM.size)); 
	# 	testGM1.display(); 
	# if(testGM2.size > numFinal):
	# 	print('Error: testGM2 size is: ' + str(testGM2.size)); 
	# 	testGM2.display(); 

	# thirdCondenseTime = time.clock(); 
	# testSimpleCondense = kmeansCondensationTesting(testSimpleCondense,k=10);
	# testSimpleCondense.normalizeWeights();  
	# thirdCondenseTime = time.clock() - thirdCondenseTime; 
	# thirdCondenseTimestr = str(thirdCondenseTime)[0:5];

	# print("Time to condense with Pure kmeans: " + str(thirdCondenseTime) + " seconds"); 

	# [x4,y4,c4] = testSimpleCondense.plot2D(vis = False); 

	# isd3 = testGMOrig.ISD(testSimpleCondense); 

	# print("The ISD with Pure kmeans: " + str(isd3)); 
	# #isd2str = str(isd2)[0:3]+'e'+str(isd2)[len(str(isd2))-3:]


	# print('Sizes:' + str(testGM.size) + ":" + str(testGM2.size) + ":" + str(testSimpleCondense.size)); 

	# fig = plt.figure()
	# ax1 = fig.add_subplot(221)
	# con1 = ax1.contourf(x1,y1,c1, cmap=plt.get_cmap('viridis'));
	# ax1.set_title('Original Mixture'); 
	# plt.colorbar(con1); 

	# minNum = 100000; 
	# maxNum = -100000;
	# for i in range(0,len(c1)):
	# 	for j in range(0,len(c1[i])):
	# 		if(c1[i][j]<minNum):
	# 			minNum = c1[i][j]; 
	# 		if(c1[i][j] > maxNum):
	# 			maxNum = c1[i][j]; 

	# ax2 = fig.add_subplot(222)
	# con2 = ax2.contourf(x2,y2,c2, cmap=plt.get_cmap('viridis'));
	# ax2.set_title('Runnalls only: ' + firstCondenseTimestr + ' seconds');
	# ax2.set_xlabel("Error with Runnalls only: " + str(isd1)); 
	# #ax2.set_xlabel("Error with Runnalls only: " + isd1str); 
	# plt.colorbar(con2,boundaries = np.linspace(minNum,0.0001,maxNum)); 


	# ax3 = fig.add_subplot(223)
	# con3 = ax3.contourf(x3,y3,c3, cmap=plt.get_cmap('viridis'));
	# ax3.set_title('Kmeans+Runnalls: ' + secondCondenseTimestr + ' seconds');
	# ax3.set_xlabel("Error with Kmeans+Runnalls: " + str(isd2)); 
	# #ax3.set_xlabel("Error with Kmeans+Runnalls: " + isd2str); 
	# plt.colorbar(con3,boundaries = np.linspace(minNum,0.0001,maxNum)); 

	# ax4 = fig.add_subplot(224)
	# con4 = ax4.contourf(x4,y4,c4, cmap=plt.get_cmap('viridis'));
	# ax4.set_title('Pure Kmeans: ' + thirdCondenseTimestr + ' seconds');
	# ax4.set_xlabel("Error with Pure Kmeans: " + str(isd3)); 
	# #ax4.set_xlabel("Error with Pure Kmeans: " + isd3str); 
	# plt.colorbar(con4,boundaries = np.linspace(minNum,0.0001,maxNum)); 


	# '''
	# ax5 = fig.add_subplot(235)
	# con5 = ax5.contourf(x2,y2,c2-c1, cmap=plt.get_cmap('viridis'));
	# ax5.set_title('Error with Runnals only, ISD=' + isd1str);
	# plt.colorbar(con5); 


	# ax6 = fig.add_subplot(236)
	# con6 = ax6.contourf(x3,y3,c3-c1, cmap=plt.get_cmap('viridis'));
	# ax6.set_title('Error with kmeans+Runnals, ISD=' + isd2str);
	# plt.colorbar(con6); 
	# '''

	# fig.suptitle("Condensation comparison with #Initial = " + str(numInit) + " and #Final = " +str(numFinal)); 

	# plt.show();

	
