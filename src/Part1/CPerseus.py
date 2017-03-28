

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



import numpy as np
from scipy.stats import multivariate_normal
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
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
#cProfile.run('re.compile("foo|bar")','restats'); 

class Perseus:


	def __init__(self,nB = 2,gamma = .95, dis = True, tag = True, beliefFileName = None):
		
		self.exitFlag = False; 
		self.discrete = dis; 
		self.tag = tag; 
		if(self.discrete):

			#get transition matrix
			#get observation matrix
			#get reward matrix
			[px,pz,r] = self.initializeOldGrid(); 
			self.px = px;
			self.pz = pz; 
			self.r = r; 

			#get initial beliefs
			self.B = self.getInitialBeliefs(nB,px);
			
			#intialize alphas
			self.Gamma = []; 
			self.Gamma += [[min(r)/(1-gamma) for i in range(0,len(px))]]; 
			self.Gamma += [[min(r)/(1-gamma) for i in range(0,len(px))]];

		else:

			if(tag == False):

				#get transition matrix
				self.buildConTransitionMatrix(); 
				#get observation matrix
				self.buildConObservationModel(); 
				#get reward matrix
				self.buildConRewardModel(); 

				#get initial beliefs
				self.buildConInitialBeliefs(nB); 
			else:
				self.buildTagTransitions(); 
				self.buildTagObservations(); 
				

				self.buildTagRewards(); 
				if(beliefFileName == None):
					self.buildTagInitialBeliefs(nB); 
					
				else:
					self.B = np.load(beliefFileName).tolist(); 




			#intialize alphas
			#each alpha is a GM
			self.Gamma = copy.deepcopy(self.r);
			#self.Gamma.addG(Gaussian([4.5,4.5,4.5,4.5],(np.eye(4)*100).tolist(),0.01)); 
			self.Gamma = [self.Gamma]; 
			
			for i in range(0,len(self.Gamma)):
				self.Gamma[i].action = 0; 
			

			
			

			self.discount = 0.95;  

		#print("Inititalization Complete");

		


	def solve(self,N = 1,nBranch = 4,verbose = False, maxMix = 20, finalMix = 20):
		
		for counter in range(0,N):
	
			if(self.exitFlag):
				break; 

			if(verbose):
				print("Iteration: " + str(counter+1)); 

			if(self.discrete):
				bestAlphas = [[0 for i in range(0,len(self.B[0]))] for j in range(0,len(self.B))]
				Value = [0 for i in range(0,len(self.B))]; 
			else:
				bestAlphas = [GM()]*len(self.B); 
				Value = [0]*len(self.B); 


			for b in self.B:
				if(self.discrete):
					bestAlphas[self.B.index(b)] = self.getMaxingAlpha(b); 
					Value[self.B.index(b)] = self.dotProduct(bestAlphas[self.B.index(b)],b); 
				else:
					bestAlphas[self.B.index(b)] = self.Gamma[np.argmax([self.continuousDot(self.Gamma[j],b) for j in range(0,len(self.Gamma))])]
					Value[self.B.index(b)] = self.continuousDot(bestAlphas[self.B.index(b)],b); 

			GammaNew = []; 


			BTilde = copy.deepcopy(self.B); 

			if(self.discrete == False):
				self.preComputeAls(); 
		
			while(len(BTilde) > 0):

				if(self.exitFlag):
					break; 

				b = random.choice(BTilde); 

				

				BTilde.remove(b); 

				if(self.discrete):
					back = self.backup(b); 
				else:
					back = self.conBackup(b); 
				al = back[0];  
				act = back[1]; 
				al.action = act; 
				
				if(self.discrete):
					if(self.dotProduct(al,b) < Value[self.B.index(b)]):
						al = bestAlphas[self.B.index(b)]; 
				else:
					if(self.continuousDot(al,b) < Value[self.findB(b)]):
						index = 0; 
						for h in self.B:
							if(b.comp(h)):
								index = self.B.index(h); 
						al = bestAlphas[index]; 

				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
					if(self.discrete):
						if(self.dotProduct(al,bprime) >= Value[self.B.index(bprime)]):
							BTilde.remove(bprime); 
					else:
						if(self.continuousDot(al,bprime) >= Value[self.findB(bprime)]):
							BTilde.remove(bprime); 

				GammaNew += [al]; 



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
						GammaNew[i].condense(max_num_mixands=maxMix);
				elif(counter == N-1):
					for i in range(0,len(GammaNew)):
						GammaNew[i].condense(max_num_mixands=finalMix);

			if(verbose and self.exitFlag == False):
				#GammaNew[0].display(); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Reduced number of mixands: " + str(av)); 
				print("Actions: " + str([GammaNew[i].action for i in range(0,len(GammaNew))])); 
				print("");

			if(self.exitFlag == False):
				self.Gamma = copy.deepcopy(GammaNew); 






	def dotProduct(self,a,b):
		value = 0; 
		for i in range(0,len(b)):
			value = value+a[i]*b[i]; 
		return value; 

	def distance(self,x1,y1,x2,y2):
		a = (x1-x2)*(x1-x2); 
		b = (y1-y2)*(y1-y2); 
		return sqrt(a+b); 


	def getMaxingAlpha(self,b):
		bestVal = -1000;
		bestInd = 0; 
		for i in self.Gamma:
			tmpVal = 0; 
			for j in i:
				if(i.index(j) >= len(b)):
					continue;
				tmpVal += b[i.index(j)]*j; 
			if(tmpVal > bestVal):
				bestVal = tmpVal; 
				bestInd = self.Gamma.index(i); 
		return self.Gamma[bestInd]; 

	def getInitialBeliefs(self,nB,px):
		#Chooses random initial beliefs
		#In the future they should be random reachable beliefs
		l = len(px); 
		B = []; 
		for i in range(0,nB):
			b = [0 for i in range(0,l)]; 
			for j in range(0,l):
				b[j] = random.random();
			b = self.normalize(b); 
			B = B + [b];  
		return B;    

	def buildTagTransitions(self):
		#so delA becomes state dependent... wonderful
		#need an array for each action, within which there is an array for the 5 possible movements of the robber
		
		self.delAVar = [[.0001,0,0,0],[0,.0001,0,0],[0,0,1,0],[0,0,0,1]]; 
		self.delA = [[0 for i in range(0,5)] for j in range(0,5)]; 

		#i is cop, j is robber, cardinal directions
		for i in range(0,5):
			for j in range(0,5):
				if(i == 0):
					if(j == 0):
						self.delA[i][j] = [-1,0,-1,0]
					elif(j == 1):
						self.delA[i][j] = [-1,0,1,0]
					elif(j == 2):
						self.delA[i][j] = [-1,0,0,1]
					elif(j == 3):
						self.delA[i][j] = [-1,0,0,-1]
					elif(j == 4):
						self.delA[i][j] = [-1,0,0,0]
				elif(i == 1):
					if(j == 0):
						self.delA[i][j] = [1,0,-1,0]
					elif(j == 1):
						self.delA[i][j] = [1,0,1,0]
					elif(j == 2):
						self.delA[i][j] = [1,0,0,1]
					elif(j == 3):
						self.delA[i][j] = [1,0,0,-1]
					elif(j == 4):
						self.delA[i][j] = [1,0,0,0]
				elif(i == 2):
					if(j == 0):
						self.delA[i][j] = [0,1,-1,0]
					elif(j == 1):
						self.delA[i][j] = [0,1,1,0]
					elif(j == 2):
						self.delA[i][j] = [0,1,0,1]
					elif(j == 3):
						self.delA[i][j] = [0,1,0,-1]
					elif(j == 4):
						self.delA[i][j] = [0,1,0,0]
				elif(i == 3):
					if(j == 0):
						self.delA[i][j] = [0,-1,-1,0]
					elif(j == 1):
						self.delA[i][j] = [0,-1,1,0]
					elif(j == 2):
						self.delA[i][j] = [0,-1,0,1]
					elif(j == 3):
						self.delA[i][j] = [0,-1,0,-1]
					elif(j == 4):
						self.delA[i][j] = [0,-1,0,0]
				elif(i == 4):
					if(j == 0):
						self.delA[i][j] = [0,0,-1,0]
					elif(j == 1):
						self.delA[i][j] = [0,0,1,0]
					elif(j == 2):
						self.delA[i][j] = [0,0,0,1]
					elif(j == 3):
						self.delA[i][j] = [0,0,0,-1]
					elif(j == 4):
						self.delA[i][j] = [0,0,0,0]
				


	def buildTagObservations(self):
		'''
		self.pz = [0]*2; 
		#if the robber is right next to the cop, have 1, otherwise zero

		self.pz[0] = GM(); 
		self.pz[1] = GM();

		sig = 5.0; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.pz[1].addG(Gaussian(i,var,1)); 

		g = Gaussian(); 
		g.mean = [4.5,4.5,4.5,4.5]; 
		g.var = (np.eye(4)*10).tolist(); 
		self.pz[0].addG(g); 
		'''


		#Build in beacon points, say at the corners and the middle for fun
		self.pz = [0]*6; 
		for i in range(0,6):
			self.pz[i] = GM(); 

		#No detects
		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		sig = 5.0; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 
		for i in range(0,5):
			g = Gaussian(); 
			g.mean = [4.5,4.5,4.5,4.5]; 
			g.var = (np.eye(4)*10).tolist(); 
			self.pz[0].addG(g); 

			
			self.pz[i].addG(Gaussian(ms[i],var,1)); 



		#Detections
		self.pz[5] = GM();
		sig = 5.0; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.pz[5].addG(Gaussian(i,var,1)); 





	def buildTagRewards(self):
		'''
		self.r = [0]*5; 

		for i in range(0,4):
			self.r[i] = GM(); 
			self.r[i].addG(Gaussian([4.5,4.5,4.5,4.5],np.eye(4)*10,0.01)); 
		self.r[4] = GM(); 

		

		sig = 0.01; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.r[4].addG(Gaussian(i,var,1)); 
		'''
		self.r = GM();  

		sig = 0.01; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.r.addG(Gaussian(i,var,1));

		'''
		var = np.eye(4)*5; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.r.addG(Gaussian(i,var,-0.5)); 
		'''


		


	def buildTagInitialBeliefs(self,nb = 20):
		
		'''
		self.B = [0]*(nb+4); 
		self.B[0] = GM(); 
		for i in range(0,100):
			g = Gaussian(); 
			g.mean = [0,0,i/10,i%10]; 
			g.weight = 1; 
			g.var = np.eye(4)*30; 
			self.B[0].addG(g); 

		self.B[1] = GM(); 
		for i in range(0,100):
			g = Gaussian(); 
			g.mean = [9,0,i/10,i%10]; 
			g.weight = 1; 
			g.var = np.eye(4)*30; 
			self.B[1].addG(g); 

		self.B[2] = GM(); 
		for i in range(0,100):
			g = Gaussian(); 
			g.mean = [0,9,i/10,i%10]; 
			g.weight = 1; 
			g.var = np.eye(4)*30; 
			self.B[2].addG(g); 

		self.B[3] = GM(); 
		for i in range(0,100):
			g = Gaussian(); 
			g.mean = [9,9,i/10,i%10]; 
			g.weight = 1; 
			g.var = np.eye(4)*30; 
			self.B[3].addG(g); 


		for i in range(4,nb+4):
			self.B[i] = GM(); 
			for j in range(0,20):
				cov = np.eye(4)*3; 
				self.B[i].addG(Gaussian([random.random()*10,random.random()*10,random.random()*10,random.random()*10],cov)); 
		'''
		
		self.B = [0]*(nb);
		for i in range(0,nb):
			self.B[i] = GM(); 
			for j in range(0,40):
				cov = np.eye(4)*random.random()*8.0; 
				self.B[i].addG(Gaussian([random.random()*10.0,random.random()*10.0,random.random()*10.0,random.random()*10.0],cov)); 
		

	

	def buildConTransitionMatrix(self):
		#Defines the mapping for each action
		self.delA = [[-1,0],[1,0],[0,1],[0,-1],[0,0]]; 
		self.delVar = [[1,0],[0,1]]; 


	def buildConObservationModel(self):
		#defined for each observation
		#for simplicity, just put a gaussian on each spot
		self.pz = [0]*100; 
		for i in range(0,100):
			self.pz[i] = GM(); 
			self.pz[i].addG(Gaussian([i/10,i%10],[[1,0],[0,1]],1)); 
		

	def buildConRewardModel(self):
		#defined as a set w,u,sig
		self.r = GM([1,1],[[9,9],[9,9]],[[[1,0],[0,1]],[[1,0],[0,1]]])


	def buildConInitialBeliefs(self,nB = 1,nMix = 5):
		#each belief is a GM
		self.B = GM([1,1],[[0,0],[1,0]],[[[1,0],[0,1]],[[1,0],[0,1]]]); 
		self.B = [self.B] + [self.B]; 
		
		self.B = [0]*(nB+1); 
		for i in range(0,nB):
			self.B[i] = GM(); 
			for j in range(0,nMix):
				self.B[i].addG(Gaussian([random.random()*10,random.random()*10],[[1,0],[0,1]],1)); 

		self.B[nB] = GM(); 
		for i in range(0,100):
			self.B[nB].addG(Gaussian([i/10,i%10],[[1,0],[0,1]],1)); 

		
		


	def PBVIArgingStep1(self,als,b):
		#returns a single alpha vector
		best = als[0];  
		bestVal = -100000; 
		for i in range(0,len(als)):
			tmpsum = 0.0; 
			for s in range(0,100):
				tmpsum = tmpsum+als[i][s]*b[s]; 
			if(tmpsum > bestVal):
				bestVal = tmpsum; 
				best = als[i]; 
		return best; 


	def extractRobMove(self,s):
		x1 = s[0]; 
		y1 = s[1]; 
		x2 = s[2]; 
		y2 = s[3];


		#Robot should always move away, with .2 chance of staying still

		if(x1 > x2):
			if(y1 > y2):
				return np.random.choice([0,2,4],p = [.4,.4,.2]); 
			elif(y2 > y1):
				return np.random.choice([0,3,4], p= [.4,.4,.2]); 
			elif(y1 == y2):
				return np.random.choice([0,4],p = [.8,.2]); 
		elif(x1 < x2):
			if(y1 > y2):
				return np.random.choice([1,2,4],p = [.4,.4,.2]); 
			elif(y2 > y1):
				return np.random.choice([1,3,4], p= [.4,.4,.2]); 
			elif(y1 == y2):
				return np.random.choice([1,4],p = [.8,.2]); 
		elif(x1 == x2):
			if(y1 > y2):
				return np.random.choice([2,4],p = [.8,.2]); 
			elif(y2 > y1):
				return np.random.choice([3,4], p= [.8,.2]); 
			elif(y1 == y2):
				return 4; 



	def preComputeAls(self):
		G = self.Gamma; 
		R = self.r; 
		pz = self.pz; 

		als1 = [[[0 for i in range(0,len(pz))] for j in range(0,len(self.delA))] for k in range(0,len(G))]; 

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				for o in range(0,len(pz)):
					als1[j][a][o] = GM(); 
					for k in range(0,G[j].size):
						for l in range(0,pz[o].size): 
							#get weights wk,wl, and del
							weight = G[j].Gs[k].weight*pz[o].Gs[l].weight*multivariate_normal.pdf(pz[o].Gs[l].mean,G[j].Gs[k].mean,(np.matrix(G[j].Gs[k].var)+np.matrix(pz[o].Gs[l].var)).tolist()); 

							#get sig and ss
							sigtmp = (np.matrix(G[j].Gs[k].var).I + np.matrix(pz[o].Gs[l].var)).tolist(); 
							sig = np.matrix(sigtmp).I.tolist(); 
						
							sstmp = np.matrix(G[j].Gs[k].var).I*np.transpose(np.matrix(G[j].Gs[k].mean)) + np.matrix(pz[o].Gs[l].var).I*np.transpose(np.matrix(pz[o].Gs[l].mean)); 
							ss = np.dot(sig,sstmp).tolist(); 

							

							if(self.tag != True):
								smean = (np.transpose(np.matrix(ss)) - np.matrix(self.delA[a])).tolist(); 
								sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist(); 
							else:
								robMove = self.extractRobMove(ss); 
								#TODO: You switched the - to a + here
								smean = (np.transpose(np.matrix(ss)) - np.matrix(self.delA[a][robMove])).tolist(); 
								sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist(); 
								
							als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight)); 
		self.preAls = als1; 


	def conBackup(self,b):
		G = self.Gamma; 
		R = self.r; 
		pz = self.pz; 

		als1 = self.preAls; 
		

		#one alpha for each belief, so one per backup

		
		bestVal = -10000000000; 
		bestAct= 0; 
		bestGM = []; 

		for a in range(0,len(self.delA)):
			suma = GM(); 
			for o in range(0,len(pz)):
				suma.addGM(als1[np.argmax([self.continuousDot(als1[j][a][o],b) for j in range(0,len(als1))])][a][o]); 
			suma.scalerMultiply(self.discount); 
			suma.addGM(R); 

			tmp = self.continuousDot(suma,b);
			#print(a,tmp); 
			if(tmp > bestVal):
				bestAct = a; 
				bestGM = suma; 
				bestVal = tmp; 


		return [bestGM,bestAct]; 
				
			


	def findB(self,b):
		for beta in self.B:
			if(beta.comp(b)):
				return self.B.index(beta); 




	def backup(self,b):

		Gamma = self.Gamma; 
		px = self.px; 
		pz = self.pz; 
		R = self.r; 

		als1 = [[[[0.0 for s in range(0,len(px))] for i in range(0,len(Gamma))] for a in range(0,len(px[0]))] for z in range(0,len(px))]; 
		
		for a in range(0,len(px[0])):
			for z in range(0,len(px)):
				for i in range(0,len(Gamma)):
					for s in range(0,len(px)):
						als1[z][a][i][s] = 0.0; 
						for sprime in range(0,len(px)):
							als1[z][a][i][s] = als1[z][a][i][s]+ .95*px[s][a][sprime]*pz[sprime][z]*Gamma[i][sprime]; 

		GammaNew = [];
		ActionsNew = [];  

		bestAction = 0; 
		bestActionVal = -10000
		bestAlpha = []; 
		als2 = [[0.0 for s in range(0,len(px))]  for a in range(0,len(px[0]))]; 
		for a in range(0,len(px[0])):
			tmpVal = 0.0; 
			
			for z in range(0,len(px)):
				tmp = self.PBVIArgingStep1(als1[z][a],b); 
				for sa in range(0,len(px)):
					als2[a][sa] = als2[a][sa] + tmp[sa]; 

			for s in range(0,len(px)):
				
				tmpR = R[s]*b[s];
				als2[a][s] = als2[a][s] + tmpR; 
				tmpVal = tmpVal + als2[a][s]; 

			if(bestActionVal < tmpVal):
				bestActionVal = tmpVal; 
				bestAction = a; 
				bestAlpha = als2[a]; 

			if(bestAlpha not in GammaNew):
				bestAlpha = bestAlpha + [bestAction]; 
				GammaNew.append(bestAlpha); 
				ActionsNew.append(bestAction); 

			#print(len(GammaNew)); 

		return [GammaNew,ActionsNew]; 

	
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



	def continuousDot(self,a,b):
		'''
		#TODO: This isn't right
		#ok so lets make a quick approximation, just for kicks...
		#go ahead and instantiate only one multivariate normal, with the average covariance? 
		base = np.matrix(np.zeros((4,4))); 
		for k in range(0,a.size):
			for l in range(0,b.size):
				base = base + (np.matrix(a.Gs[k].var) +np.matrix(b.Gs[l].var))
		s = a.size*b.size; 
		base = np.divide(base,s); 
		'''


		suma = 0; 
		for k in range(0,a.size):
			#mvn = multivariate_normal(mean = a.Gs[k].mean, cov = base.tolist());
			for l in range(0,b.size):
				suma += a.Gs[k].weight*b.Gs[l].weight*multivariate_normal.pdf(b.Gs[l].mean,a.Gs[k].mean,(np.matrix(a.Gs[k].var) +np.matrix(b.Gs[l].var)).tolist()); 
				#suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean); 


		return suma; 
				

	def printToSimpleAlphas(self,fileName = "tmpAlphas.txt"):
		f = open(fileName,"w"); 

		

	
		for i in range(0,len(self.Gamma)):
			self.Gamma[i].printToFile(f); 
		f.close(); 
		


	def initializeOldGrid(self,cor = .85, zed = 1):

		print("Initializing Grid"); 
		#specify the transition probabilities
		p = [[[0 for i in range(0,100)] for u in range(0,5)] for j in range(0,100)];  
		#cor = .99; 

		#when you wait, you don't move
		for i in range(0,100):
			for j in range(0,100):
				if(i == j):
					p[i][4][j] = 1; 
		#you can't move farther than 1 step
		for i in range(0,100):
			for j in range(0,100):
				for u in range(0,4):
					if(j != i+1 and j != i-1 and j!=i-10 and j!= i+10):
						continue;
					if(u == 0):		
						if(j==i+1):
							p[i][u][j] = cor; 
						else:
							p[i][u][j] = (1-cor)/3; 
					elif(u==1):
						if(j==i-1):
							p[i][u][j] = cor; 
						else:
							p[i][u][j] = (1-cor)/3; 
					elif(u==2):
						if(j==i-10):
							p[i][u][j] = cor; 
						else:
							p[i][u][j] = (1-cor)/3; 
					elif(u==3):
						if(j==i+10):
							p[i][u][j] = cor; 
						else:
							p[i][u][j] = (1-cor)/3; 


		#make sure you can't go around the edges...
		#for p[10][0][9] = 0
		for i in range(1,10):
			for k in range(0,5):
				p[i*10][k][(i*10)-1] = 0; 

		#p[9][1][10] = 0
		for i in range(0,9):
			for k in range(0,5):
				p[(i*10)+9][k][(i*10)+10] = 0; 


		#normalize everything
		for i in range(0,100):
			for u in range(0,5):
				tmpsum = 0; 
				for j in range(0,100):
					tmpsum = p[i][u][j] + tmpsum; 
				for j in range(0,100):
					p[i][u][j] = p[i][u][j]/tmpsum; 


		#specify the observation probabilties	
		#specify the standard deviation and covariance matrix
		variance = zed; 
		#variance = 1;
		covariance = [[variance*variance,0],[0,variance*variance]]; 
		z = [[0 for i in range(0,100)] for j in range(0,100)]; 
		for i in range(0,100):
			for j in range(0,100):
				z[i][j] = multivariate_normal.pdf([j%10,j/10],[i%10,i/10],covariance); 
				#if(i==j):
					#z[i][j] = 1; 

		for i in range(0,100):
			z[i] = self.normalize(z[i]);


		r = [-1 for i in range(0,100)]; 
		r[99] = 99; 

		return p,z,r; 

	def normalize(self,a):
		tmpsum = 0;
		b = [0 for k in range(0,len(a))];  
		for j in range(0,len(a)):
			tmpsum = tmpsum+a[j]; 
		for h in range(0,len(a)):
			b[h] = a[h]/tmpsum; 
		return b; 

	def signal_handler(self,signal, frame):
		print("Stopping Policy Generation and printing to file"); 
		self.exitFlag = True; 





def convertVectorToGrid(b):
	a = [[0 for i in range(0,10)] for j in range(0,10)]; 
	for i in range(0,100):
		a[i/10][i%10] = b[i]; 
	return a; 

if __name__ == "__main__":
	
	a = Perseus(nB = 20,dis = False, tag = True, beliefFileName = "cTagBeliefs5.npy"); 
	signal.signal(signal.SIGINT, a.signal_handler)

	a.solve(N = 80,verbose = True,maxMix = 10, finalMix = 100);
	#print(""); 
	#print("Policy Generated"); 
	#a.Gamma[0].display();   
	
	
	file = "../policies/cTagAlphas6.txt"
	file2 = "../policies/cTagAlphas6.npy"; 
	a.printToSimpleAlphas(file); 

	f = open(file2,"w"); 
	np.save(f,a.Gamma); 
	
	

	'''
	sig = 10.0; 
	ind = sig*(9.0/10.0); 
	var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 
	b = multivariate_normal.pdf([5.2,5,5.2,5],[5,5,5,5],var); 
	print(b); 
	'''


	'''
	B = GM(); 
	g = Gaussian(); 
	g.mean = [1,0,3,4]; 
	g.var = np.eye(3).tolist(); 
	g.weight = 1; 
	B.addG(g); 
	B.display(); 
	'''

	'''
	b = np.matrix([[2,3],[0,1]]); 
	c = np.matrix([[2,0],[0,2]]); 
	print(np.dot(b,4))
	print(b*4)
	'''

	
	'''
	x = [0]*100; 
	for i in range(0,100):
		x[i] = a.Gamma[np.argmax([a.continuousDot(a.Gamma[j],GM(1,[i/10,i%10],[[1,0],[1,0]])) for j in range(0,len(a.Gamma))])].action; 
	print(x); 

	plt.imshow(convertVectorToGrid(x)); 
	plt.show()
	'''


	'''
	f,axarr = plt.subplots(len(a.Gamma),sharex = True); 
	for i in range(0,len(a.Gamma)):
		axarr[i].imshow(convertVectorToGrid(a.Gamma[i])); 
	
	plt.show(); 
	'''
	

