'''
######################################################

File: testBranchAndBound.py
Author: Luke Burks
Date: April 2017

Implements the Branch and Bound algorithm in 
Kochenderfer chapter 6 on the hallway problem


######################################################
'''

from __future__ import division
from sys import path

path.append('../../src/');
from gaussianMixtures import GM, Gaussian 
from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 
from scipy.stats import norm; 
import time; 


class OnlineSolver():


	def __init__(self):
		modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass();
		self.solveMDP(); 
		self.findQ(); 
		self.UpperU = self.Q; 
		self.LowerU = self.findLowerBound();  

	def dotProduct(self,a,b):
		suma = 0; 
		for i in range(0,len(a)):
			suma+=a[i]*b[i]; 
		return suma; 


	def findLowerBound(self):
		alphaLower = [0]*self.model.acts; 
		for a in range(0,self.model.acts):
			alphaLower[a] = [min(self.model.R[a])/(1-self.model.discount)]*self.model.N; 
		for k in range(0,100):
			for a in range(0,self.model.acts):
				for s in range(0,self.model.N):
					alphaLower[a][s] = self.model.R[a][s]; 
					for sprime in range(0,self.model.N):
						alphaLower[a][s] += self.model.discount*self.model.px[a][s][sprime]*alphaLower[a][sprime];  


		for a in range(0,self.model.acts):
			for s in range(0,self.model.N):
				alphaLower[a][s] = alphaLower[a][s] -10000; 

		return alphaLower; 

	def beliefUpdate(self,bel,a,o):
		belBar = [0]*self.model.N; 
		belNew = [0]*self.model.N; 
		suma = 0; 
		for i in range(0,self.model.N):
			belBar[i] = sum(self.model.px[a][j][i]*bel[j] for j in range(0,self.model.N)); 
			belNew[i] = self.model.pz[o][i]*belBar[i];
			suma+=belNew[i];  
		#normalize
		for i in range(0,self.model.N):
			belNew[i] = belNew[i]/suma; 

		return belNew; 

	def listComp(self,a,b): 
		if(len(a) != len(b)):
			return False; 

		for i in range(0,len(a)):
			if(a[i] != b[i]):
				return False; 

		return True; 

	def normalize(self,a):
		suma = 0; 
		b=[0]*len(a); 
		for i in range(0,len(a)):
			suma+=a[i]
		for i in range(0,len(a)):
			b[i] = a[i]/suma; 
		return b; 


	def expectationOfObservation(self,b,o):
		suma = 0; 
		for i in range(0,len(b)):
			suma += b[i]*self.model.pz[o][i]; 
		return suma; 

	def solveMDP(self):
		
		self.V = [min(min(self.model.R))]*self.model.N; 
		W = [np.random.random()]*self.model.N; 

		while(not self.listComp(self.V,W)):
			W = deepcopy(self.V); 
			for i in range(0,self.model.N):
				self.V[i] = self.model.discount * max(self.model.R[a][i] + sum(self.V[j]*self.model.px[a][i][j] for j in range(0,self.model.N)) for a in range(0,self.model.acts)); 
	

	def findQ(self):
		self.Q = np.zeros(shape=(self.model.acts,self.model.N)).tolist(); 

		for a in range(0,self.model.acts):
			for i in range(0,self.model.N):
				self.Q[a][i] = self.model.R[a][i] + sum(self.V[j]*self.model.px[a][i][j] for j in range(0,self.model.N)); 


	def getLowestBound(self,bel):
		y = [0]*self.model.acts; 
		for i in range(0,len(y)):
			y[i] = self.dotProduct(bel,self.LowerU[i]); 
		return np.argmin(y); 
		


	def branchAndBound(self,bel,d):
		if(d==0):
			return [None,self.getLowestBound(bel)]
		[astar,ulower] = [None,-100000000000000]; 
		for a in range(0,self.model.acts):
			if(self.dotProduct(self.UpperU[a],bel) < ulower):
				return [astar,ulower]; 
			u = self.dotProduct(self.model.R[a],bel); 
			
			for o in range(0,self.model.obs):
				bprime = self.beliefUpdate(bel,a,o); 
				[aprime,uprimeLower] = self.branchAndBound(bprime,d-1); 
				u += self.model.discount*self.expectationOfObservation(bel,o)*uprimeLower; 
			
		

			if(u >= ulower):
				[astar,ulower] = [a,u]; 
		return [astar,ulower]; 



def testBranchAndBound():
	a = OnlineSolver();

	allActs = [-1]*a.model.N; 
	allUtils = [-100]*a.model.N;
	for i in range(0,a.model.N):
		bel = [0.001]*a.model.N; 
		bel[i] = 1; 
		bel = a.normalize(bel);
		[allActs[i],allUtils[i]] = a.branchAndBound(bel,1); 

	print(allActs); 
	
	fig,axarr = plt.subplots(2,sharex=True); 

	x = [i for i in range(0,a.model.N)]; 
 
	axarr[0].plot(x,allUtils,linewidth=5);
	axarr[0].set_title('Forward Search Approximate Utility Function'); 

	grid = np.ones(shape=(1,a.model.N)); 

	
	axarr[1].scatter(-5,.5,c='k'); 
	axarr[1].scatter(-5,.5,c='r');
	axarr[1].scatter(-5,.5,c='y');

	for i in range(0,a.model.N):
		bel = [0]*a.model.N;  
		bel[i] = 1; 
		grid[0][i] = allActs[i];
	axarr[1].set_xlim([0,20]); 
	axarr[1].imshow(grid,extent=[0,a.model.N-1,0,1],cmap='inferno');
	axarr[1].legend(['Left','Right','Stay']); 
	axarr[1].set_title('Comparison to MDP policy implementation'); 

	plt.show();



if __name__ == "__main__":

	testBranchAndBound(); 	