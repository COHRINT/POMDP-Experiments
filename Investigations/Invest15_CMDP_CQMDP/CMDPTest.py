from __future__ import division
from sys import path

path.append('../../src/');
from gaussianMixtures import GM, Gaussian 
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import numpy as np
import copy
import matplotlib.pyplot as plt







def MDPValueIteration(r,delA,delAVar):

	#Intialize Value function
	ValueFunc = copy.deepcopy(r[4]); 

	'''
	for g in ValueFunc.Gs:
		g.weight = -1000; 
	'''
	#ValueFunc.scalerMultiply(1/(.1)); 

	comparision = GM(); 
	comparision.addG(Gaussian([1,0],[[1,0],[0,1]],1)); 

	uniform = GM(); 
	for i in range(0,5):
		for j in range(0,5):
			uniform.addG(Gaussian([i,j],[[2,0],[0,2]],1)); 

	count = 0; 

	#until convergence
	#TODO: Should be count < 100ish
	while(not ValueFunc.fullComp(comparision) and count < 100):
		print(count); 
		comparision = copy.deepcopy(ValueFunc); 
		count += 1;
		#print(count); 
		maxVal = -10000000; 
		maxGM = GM(); 
		for a in range(0,2):
			suma = GM(); 
			for g in ValueFunc.Gs:
				mean = (np.matrix(g.mean)-np.matrix(delA[a])).tolist(); 
				var = (np.matrix(g.var) + np.matrix(delAVar)).tolist();
				suma.addG(Gaussian(mean,var,g.weight));  
			suma.addGM(r[a]); 
			tmpVal = continuousDot(uniform,suma); 
			if(tmpVal > maxVal):
				maxVal = tmpVal; 
				maxGM = copy.deepcopy(suma); 

		#maxGM.scalerMultiply(.9);

		[x,y,c] = maxGM.plot2D(low=[0,0],high=[5,5],title='Value',vis=False); 
		plt.contourf(x,y,c); 
		plt.pause(1);

		#maxGM = maxGM.kmeansCondensationN(10); 
		maxGM.condense(5)

		ValueFunc = copy.deepcopy(maxGM); 

	return ValueFunc; 


def getMDPAction(self,x):
	maxVal = -10000000; 
	maxGM = GM();
	bestAct = 0;  
	for a in range(0,len(self.delA)):
		suma = GM(); 
		for g in self.ValueFunc.Gs:
			mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist(); 
			var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
			suma.addG(Gaussian(mean,var,g.weight));  
		suma.addGM(self.r); 
		
		tmpVal = suma.pointEval(x); 
		if(tmpVal > maxVal):
			maxVal = tmpVal; 
			maxGM = suma;
			bestAct = a; 
	return bestAct; 

def solveQ(self):

	self.Q =[0]*len(self.delA); 
	V = self.ValueFunc; 
	for a in range(0,len(self.delA)):
		self.Q[a] = GM(); 
		for i in range(0,V.size):
			mean = (np.matrix(V.Gs[i].mean)-np.matrix(self.delA[a])).tolist(); 
			var = (np.matrix(V.Gs[i].var) + np.matrix(self.delAVar)).tolist()
			self.Q[a].addG(Gaussian(mean,var,V.Gs[i].weight)); 
		self.Q[a].addGM(self.r); 
	return Q; 

def getQMDPAction(self,b):
	act = np.argmax([self.continuousDot(self.Q[j],b) for j in range(0,len(self.Q))]);
	return act; 

def continuousDot(a,b):
	suma = 0;  

	if(isinstance(a,np.ndarray)):
		a = a.tolist(); 
		a = a[0]; 

	if(isinstance(a,list)):
		a = a[0];

	a.clean(); 
	b.clean(); 

	for k in range(0,a.size):
		for l in range(0,b.size):
			suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean, np.matrix(a.Gs[k].var)+np.matrix(b.Gs[l].var)); 
	return suma; 



def testCMDP():
	delA = [[-0.5,0],[.5,0],[0,.5],[0,-.5],[0,0]]; 
	delAVar = [[0.5,0],[0,0.5]]; 
	r = [0]*5; 
	for i in range(0,5):
		r[i] = GM();
		m = (np.array([4.5,.5])-np.array(delA[i])).tolist(); 
		r[i].addG(Gaussian(m,[[.25,0],[0,.25]],5));    
		
		
		for x in range(-2,8):
			for y in range(-2,8):
				r[i].addG(Gaussian([x,y],[[1,0],[0,1]],-2)); 
		

		#r[i].condense(20); 
		#r[i].plot2D(low=[0,0],high=[5,5],title='Reward');

	Value = MDPValueIteration(r,delA,delAVar)


def testCQMDP():
	pass; 


if __name__ == "__main__":
	testCMDP(); 
	testCQMDP(); 