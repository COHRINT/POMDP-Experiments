from __future__ import division
'''
************************************************************************************************************************************************************
File: PolicyGeneratorSoftmax.py
Written By: Luke Burks
December 2016

This is intended as a template for POMDP policy 
generators. Ideally all problem specific bits
will have been removed

Input: -n <problemName> -b <initialBeliefNumber> -a <alphaSaveNumber> -m <maxNumMixands> -f <finalNumMixands> -g <generateNewModels>
Output: solve function
<problemName>Alphas<alphaSaveNumber>.npy, a file containing the policy found by the generator
************************************************************************************************************************************************************
'''



__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"


import numpy as np
from scipy.stats import multivariate_normal as mvn
import random
import copy
import cProfile
import re
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os; 
from math import sqrt
import signal
import sys
import cProfile
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import matplotlib.animation as animation
from numpy import arange
import time
import matplotlib.image as mgimg


class PolicyGeneratorSoftmax:

	def __init__(self,fileNamePrefix,beliefFile = None,gen=False,qGen = True):
	

		#Initialize exit flag
		self.exitFlag = False; 
		self.b = None; 

		#Grab Modeling Code
		allMod = D2DiffsModelSoftmax(fileNamePrefix); 

		#Build Transition Model
		allMod.buildTransition(); 
		self.delA = allMod.delA; 
		self.delAVar = allMod.delAVar; 

		#Build Observation Model
		if(gen==True):
			print("Building Observation Models"); 
		allMod.buildObs(gen=gen);
		self.pz = allMod.pz;
		
		#Build Reward Model
		if(gen == True):
			print("Building Reward Model"); 
		allMod.buildReward(gen=gen); 
		self.r = allMod.r; 
		self.discount = allMod.discount; 
	
		#QMDP Generation
		if(qGen == True):
			allMod.MDPValueIteration(False); 
			self.ValueFunc = allMod.ValueFunc; 
			self.solveQ(); 

		#Loading Beliefs
		if(beliefFile is None):
			print("No belief file"); 
			raise;
		self.B = np.load(beliefFile).tolist(); 
			
		#Initialize Gamma
		
		self.Gamma = copy.deepcopy(self.r); 
		for i in range(0,len(self.Gamma)):
			self.Gamma[i].action = i; 
			for g in self.Gamma[i].Gs:
				g.weight = g.weight/(1-self.discount);
			self.Gamma[i] = self.Gamma[i].kmeansCondensationN(k=20);  
		
		
		'''
		self.Gamma = [GM()]; 
		self.Gamma[0].addG(Gaussian([2.5,2.5],[[10000,0],[0,10000]],-100000000))
		'''

	
		 



	def solve(self,N,maxMix = 20, finalMix = 50, verbose = False, alComb = False, alsave = "interceptAlphasTemp.npy", visualize = False):


		for counter in range(0,N):
	
			if(self.exitFlag):
				break; 

			if(verbose):
				print("Iteration: " + str(counter+1)); 
			else:
				print("Iteration: " + str(counter+1)); 
			
			bestAlphas = [GM()]*len(self.B); 
			Value = [0]*len(self.B); 

			for b in self.B:
				bestAlphas[self.B.index(b)] = self.Gamma[np.argmax([self.continuousDot(self.Gamma[j],b) for j in range(0,len(self.Gamma))])];
				Value[self.B.index(b)] = self.continuousDot(bestAlphas[self.B.index(b)],b); 
				
			GammaNew = [];

			BTilde = copy.deepcopy(self.B); 
			 
			self.preComputeAls(); 

			while(len(BTilde) > 0):

				if(self.exitFlag):
					break; 

				b = random.choice(BTilde); 

				BTilde.remove(b); 

				al = self.backup(b); 

			
				if(self.continuousDot(al,b) < Value[self.findB(b)]):
					index = 0; 
					for h in self.B:
						if(b.fullComp(h)):
							index = self.B.index(h); 
					al = bestAlphas[index]; 
				else:
					index = 0; 
					for h in self.B:
						if(b.fullComp(h)):
							index = self.B.index(h);
					bestAlphas[index] = al; 

				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
					if(self.continuousDot(al,bprime) >= Value[self.findB(bprime)]):
						BTilde.remove(bprime); 



				#make sure the alpha doesn't already exist
				addFlag = True; 
				for i in range(0,len(GammaNew)):
					if(al.fullComp(GammaNew[i])):
						addFlag = False; 
				if(addFlag):
					GammaNew += [al];


			if(alComb):
				GammaNew = self.alphaActionComb(GammaNew); 
			
			
			if(verbose and self.exitFlag == False):
				print("Number of Alphas: " + str(len(GammaNew))); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Average number of mixands: " + str(av)); 
			if(self.exitFlag == False):
				if(counter < N-1):
					for i in range(0,len(GammaNew)):
						#if(GammaNew[i].size > maxMix):
							#GammaNew[i].condense(max_num_mixands=maxMix);
						GammaNew[i] = GammaNew[i].kmeansCondensationN(k = maxMix); 
				elif(counter == N-1):
					for i in range(0,len(GammaNew)):
						#GammaNew[i].condense(max_num_mixands=finalMix);
						GammaNew[i] = GammaNew[i].kmeansCondensationN(k = finalMix); 

			if(verbose and self.exitFlag == False):
				#GammaNew[0].display(); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Reduced number of mixands: " + str(av)); 
				print("Actions: " + str([GammaNew[i].action for i in range(0,len(GammaNew))])); 
				print("");

			if(visualize):
				fig,axarr = plt.subplots(len(GammaNew)); 

				for i in range(0,len(GammaNew)):
					[x,y,c] = GammaNew[i].plot2D(vis = False); 
					axarr[i].contourf(x,y,c,cmap = 'viridis'); 
					axarr[i].set_title('Action: '+ str(GammaNew[i].action)); 
				plt.pause(0.01); 


			if(self.exitFlag == False):
				f = open(alsave,"w"); 
				self.Gamma = copy.deepcopy(GammaNew); 
				np.save(f,self.Gamma); 
				f.close(); 
				


		f = open(alsave,"w"); 
		np.save(f,self.Gamma); 
		f.close(); 

	def alphaActionComb(self,GammaNew):
		#Test the results from just combining all the alphas into one big one for each action
		#WARNING: Not currently working...
		tmp = [0]*len(self.delA); 
		gamcount = [0]*len(self.delA); 

		for i in range(0,len(tmp)):
			tmp[i] = GM(); 
			tmp[i].action = i; 


		for g in GammaNew: 
			tmp[g.action].addGM(g);
			gamcount[g.action] = gamcount[g.action] + 1;  

		
		dels = []; 

		for i in range(0,len(tmp)):
			if(gamcount[i] != 0):
				tmp[i].scalerMultiply(1/gamcount[i]);
			else:
				dels.append(tmp[i]); 

		for g in tmp:
			if(g.size > 0):
				print(g.action); 

		print('');

		for rem in dels:
			if(rem in tmp):
				tmp.remove(rem);

		for g in GammaNew:
			print(g.action); 

		print(''); 

		for g in tmp:
			print(g.action); 

		return tmp;

	def preComputeAls(self):
		G = self.Gamma;  

		als1 = [[[0 for i in range(0,self.pz.size)] for j in range(0,len(self.delA))] for k in range(0,len(G))]; 

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				for o in range(0,self.pz.size):
					als1[j][a][o] = GM(); 

					#alObs = G[j].runVB(self.soft_weight,self.soft_bias,self.soft_alpha,self.soft_zeta_c,softClassNum = o); 
					alObs = self.pz.runVBND(G[j],o); 

					for k in alObs.Gs:
						mean = (np.matrix(k.mean) - np.matrix(self.delA[a])).tolist(); 
						var = (np.matrix(k.var) + np.matrix(self.delAVar)).tolist(); 
						weight = k.weight; 
						als1[j][a][o].addG(Gaussian(mean,var,weight)); 

		self.preAls = als1; 



	def backup(self,b):
		G = self.Gamma; 
		R = self.r; 
		pz = self.pz; 

		als1 = self.preAls; 
		

		bestVal = -10000000000; 
		bestAct= 0; 
		bestGM = []; 

		for a in range(0,len(self.delA)):
			suma = GM(); 
			for o in range(0,pz.size):
				suma.addGM(als1[np.argmax([self.continuousDot(als1[j][a][o],b) for j in range(0,len(als1))])][a][o]); 
			suma.scalerMultiply(self.discount); 
			suma.addGM(R[a]); 

			tmp = self.continuousDot(suma,b);
			#print(a,tmp); 
			if(tmp > bestVal):
				bestAct = a; 
				bestGM = copy.deepcopy(suma); 
				bestVal = tmp; 

		bestGM.action = bestAct; 

		return bestGM;  


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
		f = open("../policies/qmdpPolicyWalls.npy","w"); 
		np.save(f,self.Q);

	def getQMDPAction(self,b):
		act = np.argmax([self.continuousDot(self.Q[j],b) for j in range(0,len(self.Q))]);
		return act; 

	def getQMDPSecondaryAction(self,b,exclude=[]):
		sG = [];
		for a in range(0,len(self.delA)):
			if(a not in exclude):
				sG.append(a); 
		bestVal = -10000000000;
		act = -1; 
		for a in sG:
			tmpVal = self.continuousDot(self.Q[a],b); 
			if(tmpVal > bestVal):
				bestVal = tmpVal; 
				act = a; 
		return act; 


	def covAdd(self,a,b):
		if(type(b) is not list):
			b = b.tolist(); 
		if(type(a) is not list):
			a = a.tolist(); 

		c = copy.deepcopy(a);

		for i in range(0,len(a)):
			for j in range(0,len(a[i])):
				c[i][j] += b[i][j]; 
		return c;  



	def findB(self,b):
		for beta in self.B:
			if(beta.fullComp(b)):
				return self.B.index(beta); 


	def continuousDot(self,a,b):
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


	def signal_handler(self,signal, frame):
		print("Stopping Policiy Generation and printing to file"); 
		self.exitFlag = True; 




if __name__ == "__main__":
	



	#Files
	belLoad = '../beliefs/DiffsPolicyBeliefs2.npy';
	alsave = '../policies/DiffsSoftmaxAlphas1.npy'; 

	'''	
	********
	Alphas:
	1: 
	2: Keep
	********
	'''


	#Flips and switches

	#Solver Params
	sol = True;
	iterations = 500; 
	alphaCombination = False
	maxMix = 10; 
	finalMix = 100; 


	#controls obs and reward generation
	generate = False; 
	qGen = False


	fileNamePrefix = 'D2DiffsModelSoftmax';

	a = PolicyGeneratorSoftmax(fileNamePrefix = fileNamePrefix,beliefFile = belLoad,gen = generate,qGen = qGen); 
	signal.signal(signal.SIGINT, a.signal_handler);
	
	
	a.solve(N = iterations,maxMix = maxMix, finalMix = finalMix,alComb = alphaCombination,alsave = alsave,verbose = True,visualize = False); 
	