

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



import numpy as np
from scipy.stats import multivariate_normal
import random
import copy
import cProfile
import re
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os; 
from math import sqrt
#cProfile.run('re.compile("foo|bar")','restats'); 

class Perseus:


	def __init__(self,nB = 2,gamma = .95, dis = True, tag = True):
		
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
				self.buildTagInitialBeliefs(nB); 


			#intialize alphas
			#each alpha is a GM
			self.Gamma = copy.deepcopy(self.r);
			self.Gamma.action = 0; 
			self.Gamma = [self.Gamma] + [self.Gamma]; 

			
			

			self.discount = 0.95;  

		#print("Inititalization Complete");

		


	def solve(self,N = 1,nBranch = 4,verbose = False, maxMix = 20, finalMix = 20):
		
		for counter in range(0,N):
	
			if(verbose):
				print(counter); 

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

			if(verbose):
				print("Initial Step Complete"); 

			while(len(BTilde) > 0):

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

			if(verbose):
				print("Second Step Complete");
				print("Number of Alphas: " + str(len(GammaNew))); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Average number of mixands: " + str(av)); 



			if(counter < N-1):
				for i in range(0,len(GammaNew)):
					GammaNew[i].condense(maxMix);
			elif(counter == N-1):
				for i in range(0,len(GammaNew)):
					GammaNew[i].condense(finalMix);

			if(verbose):
				#GammaNew[0].display(); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Reduced number of mixands: " + str(av)); 

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
		
		self.delAVar = [[.01,0,0,0],[0,.01,0,0],[0,0,1,0],[0,0,0,1]]; 
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
		self.pz = [0]*2; 
		#if the robber is right next to the cop, have 1, otherwise zero

		self.pz[0] = GM(); 
		self.pz[1] = GM();

		sig = 10.0; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.pz[1].addG(Gaussian(i,var,1)); 

		g = Gaussian(); 
		g.mean = [4.5,4.5,4.5,4.5]; 
		g.var = (np.eye(4)*100).tolist(); 
		self.pz[0].addG(g); 




	def buildTagRewards(self):
		self.r = GM(); 

		sig = 10.0; 
		ind = sig*(9.0/10.0); 
		var = [[sig,0,ind,0],[0,sig,0,ind],[ind,0,sig,0],[0,ind,0,sig]]; 

		ms = [[0,0,0,0],[0,9,0,9],[9,0,9,0],[9,9,9,9],[4.5,4.5,4.5,4.5]]; 
		for i in ms:
			self.r.addG(Gaussian(i,var,1)); 

		


	def buildTagInitialBeliefs(self,nb = 20):
		self.B = [0]*(nb+1); 
		self.B[0] = GM(); 
		for i in range(0,100):
			g = Gaussian(); 
			g.mean = [0,0,i/10,i%10]; 
			g.weight = 1; 
			g.var = np.eye(4)*30; 
			self.B[0].addG(g); 

		for i in range(1,nb+1):
			self.B[i] = GM(); 
			for j in range(0,20):
				cov = np.eye(4)*3; 
				self.B[i].addG(Gaussian([random.random()*10,random.random()*10,random.random()*10,random.random()*10],cov)); 



	

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




	def conBackup(self,b):
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
							weight = G[j].Gs[k].weight*pz[o].Gs[l].weight*multivariate_normal.pdf(pz[o].Gs[l].mean,G[j].Gs[k].mean,self.covAdd(G[j].Gs[k].var,pz[o].Gs[l].var)); 

							#get sig and ss
							sigtmp = self.covAdd(np.matrix(G[j].Gs[k].var).I.tolist(), np.matrix(pz[o].Gs[l].var).tolist()); 
							sig = np.matrix(sigtmp).I.tolist(); 
						
							sstmp = np.matrix(G[j].Gs[k].var).I*np.transpose(np.matrix(G[j].Gs[k].mean)) + np.matrix(pz[o].Gs[l].var).I*np.transpose(np.matrix(pz[o].Gs[l].mean)); 
							ss = np.dot(sig,sstmp).tolist(); 

							

							if(self.tag != True):
								smean = (np.transpose(np.matrix(ss)) - np.matrix(self.delA[a])).tolist(); 
								sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist(); 
							else:
								robMove = self.extractRobMove(ss); 
								smean = (np.transpose(np.matrix(ss)) - np.matrix(self.delA[a][robMove])).tolist(); 
								sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist(); 
							
							als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight)); 

		GammaNew = []; 


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

		suma = 0; 
		for k in range(0,a.size-1):
			for l in range(0,b.size):
				suma += a.Gs[k].weight*b.Gs[l].weight*multivariate_normal.pdf(b.Gs[l].mean,a.Gs[k].mean,self.covAdd(a.Gs[k].var,b.Gs[l].var)); 
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

class Gaussian:
	def __init__(self,u = None,sig = None,w=1):
		if(u == None):
			self.mean = [0,0]; 
		else:
			self.mean = u; 
		if(sig == None):
			self.sig = [[1,0],[0,1]];
		else:
			self.var = sig; 
		self.weight = w; 
class GM:
	def __init__(self,w=None,u=None,s=None):
		self.Gs = []; 
		if(w == None):
			self.size = 0; 
		elif(isinstance(w,float) or isinstance(w,int)):
			self.size = 0; 
			self.addG(Gaussian(u,s,w)); 
		elif(len(w) > 1):
			for i in range(0,len(w)):
				self.Gs += [Gaussian(u[i],s[i],w[i])];
		self.size = len(self.Gs);  
		self.action = None; 


	def normalizeWeights(self):
		suma = 0; 
		for g in self.Gs:
			suma += g.weight; 
		for g in self.Gs:
			g.weight = g.weight/suma; 
		self.size = len(self.Gs); 

	def addGM(self,b):
		for i in range(0,len(b.Gs)):
			self.addG(b.Gs[i]);
		self.size = len(self.Gs); 

	def addG(self,b):
		self.Gs += [b];
		self.size+=1; 
		self.size = len(self.Gs); 

	def display(self):
		print("Means"); 
		print([self.Gs[i].mean for i in range(0,self.size)]); 
		print("Variances"); 
		print([self.Gs[i].var for i in range(0,self.size)]); 
		print("Weights"); 
		print([self.Gs[i].weight for i in range(0,self.size)]); 
		if(self.action is not None):
			print("Action"); 
			print(self.action); 

	def comp(self,b):
		if(self.size != b.size):
			return False; 

		for g in range(0,self.size):
			if(self.Gs[g].mean != b.Gs[g].mean):
				return False; 
			if(self.Gs[g].weight != b.Gs[g].weight):
				return False; 
			for i in self.Gs[g].var:
				if(i not in b.Gs[g].var):
					return False; 
					'''
			if(self.Gs[g].var != b.Gs[g].var):
				return False; 
				'''

		return True; 

	def printClean(self,slices):
		slices = str(slices); 
		slices = slices.replace(']',''); 
		slices = slices.replace(',','');
		slices = slices.replace('[',''); 
		return slices;

	def printToFile(self,file):
		#first line is N, number of gaussians
		#next N lines are, mean, variance, weight
		file.write(str(self.size) + " " + str(self.action) + "\n"); 
		for g in self.Gs:
			m = self.printClean(g.mean); 
			var = self.printClean(g.var); 
			w = self.printClean(g.weight); 
			file.write(m + " " + var + " " + w + "\n"); 



	def scalerMultiply(self,s):
		for g in self.Gs:
			g.weight = s*g.weight; 

	def condense(self, max_num_mixands=None):
       
		if max_num_mixands is None:
			max_num_mixands = self.max_num_mixands

		# Check if merging is useful
		if self.size <= max_num_mixands:
		    return


		# Create lower-triangle of dissimilarity matrix B
		#<>TODO: this is O(n ** 2) and very slow. Speed it up! parallelize?
		B = np.zeros((self.size, self.size))
	 
		for i in range(self.size):
		    mix_i = (self.Gs[i].weight, self.Gs[i].mean, self.Gs[i].var) 
		    for j in range(i):
		        if i == j:
		            continue
		        mix_j = (self.Gs[j].weight, self.Gs[j].mean, self.Gs[j].var) 
		        B[i,j] = self.mixand_dissimilarity(mix_i, mix_j)
		       	

		# Keep merging until we get the right number of mixands
		deleted_mixands = []
		toRemove = []; 
		while self.size > max_num_mixands:
		    # Find most similar mixands
		   
			 
			min_B = B[B>0].min()

			ind = np.where(B==min_B)
			i, j = ind[0][0], ind[1][0]

			# Get merged mixand
			mix_i = (self.Gs[i].weight, self.Gs[i].mean, self.Gs[i].var) 
			mix_j = (self.Gs[j].weight, self.Gs[j].mean, self.Gs[j].var) 
			w_ij, mu_ij, P_ij = self.merge_mixands(mix_i, mix_j)

			# Replace mixand i with merged mixand
			ij = i
			self.Gs[ij].weight = w_ij
			self.Gs[ij].mean = mu_ij.tolist(); 
			self.Gs[ij].var = P_ij.tolist(); 



			# Fill mixand i's B values with new mixand's B values
			mix_ij = (w_ij, mu_ij, P_ij)
			deleted_mixands.append(j)
			toRemove.append(self.Gs[j]);

			#print(B.shape[0]); 

			for k in range(0,B.shape[0]):
			    if k == ij or k in deleted_mixands:
			        continue

			    # Only fill lower triangle
			   # print(self.size,k)
			    mix_k = (self.Gs[k].weight, self.Gs[k].mean, self.Gs[k].var) 
			    if k < i:
			        B[ij,k] = self.mixand_dissimilarity(mix_k, mix_ij)
			    else:
			        B[k,ij] = self.mixand_dissimilarity(mix_k, mix_ij)

			# Remove mixand j from B
			B[j,:] = np.inf
			B[:,j] = np.inf
			self.size -= 1


		# Delete removed mixands from parameter arrays
		for rem in toRemove:
			self.Gs.remove(rem); 

		

	def mixand_dissimilarity(self,mix_i, mix_j):
	    """Calculate KL descriminiation-based dissimilarity between mixands.
	    """
	    # Get covariance of moment-preserving merge
	    w_i, mu_i, P_i = mix_i
	    w_j, mu_j, P_j = mix_j
	    _, _, P_ij = self.merge_mixands(mix_i, mix_j)

	    # Use slogdet to prevent over/underflow
	    _, logdet_P_ij = np.linalg.slogdet(P_ij)
	    _, logdet_P_i = np.linalg.slogdet(P_i)
	    _, logdet_P_j = np.linalg.slogdet(P_j)
	    
	    # <>TODO: check to see if anything's happening upstream
	    if np.isinf(logdet_P_ij):
	        logdet_P_ij = 0
	    if np.isinf(logdet_P_i):
	        logdet_P_i = 0
	    if np.isinf(logdet_P_j):
	        logdet_P_j = 0

	    b = 0.5 * ((w_i + w_j) * logdet_P_ij - w_i * logdet_P_i - w_j * logdet_P_j)
	    return b

	def merge_mixands(self,mix_i, mix_j):
	    """Use moment-preserving merge (0th, 1st, 2nd moments) to combine mixands.
	    """
	    # Unpack mixands
	    w_i, mu_i, P_i = mix_i
	    w_j, mu_j, P_j = mix_j

	    mu_i = np.array(mu_i); 
	    mu_j = np.array(mu_j); 

	    P_j = np.matrix(P_j); 
	    P_i = np.matrix(P_i); 

	    # Merge weights
	    w_ij = w_i + w_j
	    w_i_ij = w_i / (w_i + w_j)
	    w_j_ij = w_j / (w_i + w_j)

	    # Merge means

	    mu_ij = w_i_ij * mu_i + w_j_ij * mu_j

	    P_j = np.matrix(P_j); 
	    P_i = np.matrix(P_i); 


	    # Merge covariances
	    P_ij = w_i_ij * P_i + w_j_ij * P_j + \
	        w_i_ij * w_j_ij * np.outer(self.subMu(mu_i,mu_j), self.subMu(mu_i,mu_j))



	    return w_ij, mu_ij, P_ij

	def subMu(self,a,b):

		if(len(a) == 1):
			return a-b; 

		c = [0]*len(a); 
		for i in range(0,len(a)):
			c[i] = a[i]-b[i]; 
		return c; 


def convertVectorToGrid(b):
	a = [[0 for i in range(0,10)] for j in range(0,10)]; 
	for i in range(0,100):
		a[i/10][i%10] = b[i]; 
	return a; 

if __name__ == "__main__":
	a = Perseus(nB = 5,dis = False, tag = True); 
	a.solve(N = 10,verbose = True,maxMix = 50, finalMix = 100);
	#print(""); 
	#print("Policy Generated"); 
	a.Gamma[0].display();   
	
	file = "cTagAlphas2.txt"
	file2 = "cTagAlphas2.npy"; 
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
	

