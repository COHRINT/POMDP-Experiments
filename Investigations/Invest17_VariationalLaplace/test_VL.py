
'''
****************************************************
File: test_VL.py
Author: Luke Burks
May 2017


Testing for the Variational Laplace method of 
approximating gaussian/softmax products
 
****************************************************
'''
from __future__ import division



__author__ = "Luke Burks"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Luke Burks", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"


from sys import path
path.append('../../src'); 
from gaussianMixtures import GM
from gaussianMixtures import Gaussian 
from softmaxModels import Softmax
import numpy as np 
import matplotlib.pyplot as plt


def likeDiv(like,softClass,x):
	weights = like.weights; 
	bias = like.bias; 
	num1 = weights[softClass]*np.exp(weights[softClass]*x + bias[softClass]); 
	dem = 0; 
	num22 = 0; 
	for i in range(0,len(weights)):
		dem += np.exp(weights[i]*x + bias[i]); 
		num22 += weights[i]*np.exp(weights[i]*x + bias[i]); 
	ans = num1/dem; 
	num2 = np.exp(weights[softClass]*x + bias[softClass])*num22; 
	ans = ans + num2/(dem**2); 

	return ans; 



def VL(pri,like,softClass,low,high,res):
	[xs,softmax] = like.plot1D(low,high,res,vis = False);
	ys = [0]*len(xs); 
	for i in range(0,len(xs)):
		ys[i] = np.log(pri.pointEval(xs[i])) + np.log(softmax[softClass][i]); 
	newMean = xs[ys.index(max(ys))];
	
	newVar = -(pri.getVars()[0])**(-1);
	secDiv = likeDiv(like,softClass,newMean); 
	newVar = newVar - 1/(softmax[softClass][ys.index(max(ys))])**2 * (secDiv*secDiv); 
	newVar = -(newVar)**(-1); 


	ans = GM(); 
	ans.addG(Gaussian(newMean,newVar,pri.getWeights()[0])); 
	return ans;  




def testVL():


	#plotting parameters
	low = 0; 
	high = 5; 
	res = 100; 

	#Define Likelihood Model
	weight = [-30,-20,-10,0]; 
	bias = [60,50,30,0]; 
	softClass = 1;
	likelihood = Softmax(weight,bias); 

	#Define Prior
	prior = GM(); 
	prior.addG(Gaussian(3,0.25,1)); 

	postVL = VL(prior,likelihood,softClass,low,high,res); 
	postVB = likelihood.runVB(prior,softClassNum = softClass);
	 
	#Normalize postVB
	#postVB.normalizeWeights(); 

	#share weights
	postVL[0].weight = postVB[0].weight; 


	[x0,classes] = likelihood.plot1D(res = res,vis = False); 
	[x1,numApprox] = likelihood.numericalProduct(prior,softClass,low=low,high=high,res = res,vis= False); 
	
	softClassLabels = ['Far left','Left','Far Right','Right']; 
	labels = ['likelihood','prior','VB Posterior','VL Posterior','Numerical Posterior']; 
	[x2,pri] = prior.plot(low = low, high = high,num = res,vis = False);
	[x3,pos] = postVB.plot(low = low, high = high,num = res,vis = False); 
	[x4,pos2] = postVL.plot(low=low,high=high,num=res,vis=False); 
	plt.plot(x0,classes[softClass]); 
	plt.plot(x2,pri);
	plt.plot(x3,pos); 
	plt.plot(x4,pos2); 
	plt.plot(x1,numApprox); 
	plt.ylim([0,1.1])
	plt.xlim([low,high])
	plt.title("Fusion of prior with: " + softClassLabels[softClass]); 
	plt.legend(labels); 
	plt.show(); 




if __name__ == "__main__":
	testVL(); 









