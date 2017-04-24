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

	def approximateValueFunction(self,bel):

		U = deepcopy(self.Q[2]); 
		suma = 0; 
		for i in range(0,len(bel)):
			suma += U[i]*bel[i]; 

		return suma; 

	def expectedReward(self,b,a):
		suma = 0; 
		for i in range(0,len(b)):
			suma+=b[i]*self.model.R[a][i]; 
		return suma; 

	def expectationOfObservation(self,b,o):
		suma = 0; 
		for i in range(0,len(b)):
			suma += b[i]*self.model.pz[o][i]; 
		return suma; 


	def forwardSearch(self,bel,d):
		if(d==0):
			return (None,self.approximateValueFunction(bel)); 
		[astar,ustar] = [None,-100000000000000000000]; 

		for a in range(0,self.model.acts):
			u = self.expectedReward(bel,a); 
			for o in range(0,self.model.obs):
				bprime = self.beliefUpdate(bel,a,o); 
				[aprime,uprime] = self.forwardSearch(bprime,d-1); 
				u = u+self.model.discount*self.expectationOfObservation(bel,o)*uprime; 
			if(u>ustar):
				[astar,ustar] = [a,u]; 
		return [astar,ustar]; 

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




def testForwardSearch():
	a = OnlineSolver();

	allActs = [-1]*a.model.N; 
	allUtils = [-100]*a.model.N;
	for i in range(0,a.model.N):
		bel = [0.001]*a.model.N; 
		bel[i] = 1; 
		bel = a.normalize(bel);
		[allActs[i],allUtils[i]] = a.forwardSearch(bel,3); 

	
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

	testForwardSearch(); 