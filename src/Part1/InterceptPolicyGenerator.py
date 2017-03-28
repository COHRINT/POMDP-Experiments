from __future__ import division
'''
****************************************************
File: InterceptPolicyGenerator.py
Written By: Luke Burks
August 2016


****************************************************
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


#TODO: Fix backup
#TODO: build precompute
#TODO: Build all models
#TODO: build intializations


#TODO: You changed so that actions left and right have no reward function in the backup

class InterceptPolicyGenerator:

	def __init__(self,nb = 20,order = 0, kalman = True, beliefFile = None):
		#All the kalman matrixes

		if(kalman):
			self.initializeKalman(); 
		else:
			self.initializeMixed(); 
		 

		#Initialize exit flag
		self.exitFlag = False; 

		#Initialize beliefs
		if(beliefFile == None and kalman):
			self.B = [0]*nb; 
			for i in range(0,nb):
				self.B[i] = GM(); 
				for j in range(0,10):
					#self.B[i].addG(Gaussian([random.random()*10 - 5,0],[[1,0],[0,1]],1));
					self.B[i].addG(Gaussian(random.random()*40 - 20,1,1)); 
		elif(beliefFile == None and not kalman):
			self.B = [0]*50; 
			for i in range(0,50):
				self.B[i] = [i/10,GM()]; 
				
				self.B[i][1].addG(Gaussian(15,1,1))
				self.B[i][1].addG(Gaussian(15,1,1))
		else:
			self.B = np.load(beliefFile).tolist(); 
			


		self.buildTransition(kalman); 
		self.buildReward(kalman); 
		self.discount = 0.95; 

		#Initialize Gamma
		self.Gamma = copy.deepcopy(self.r); 
		self.Gamma = [0]*51; 
		for i in range(0,51):
			self.Gamma[i] = [GM()]; 
			self.Gamma[i][0].addG(Gaussian(2.5,100,-200)); 
			self.Gamma[i][0].action = 1; 

		'''
		if(kalman):
			for i in range(0,len(self.Gamma)):
				self.Gamma[i].action = i; 
		else:
			for i in range(0,len(self.Gamma)):
				self.Gamma[i] = [self.Gamma[i]]; 


			for x in range(0,51):
				for a in range(0,len(self.Gamma[x])): 
					self.Gamma[x][a].action = 2; 
		'''




	#TODO: rounding in the for b in self.B on mixed
	def solve(self,N,maxMix = 3, finalMix = 3,kalman = True, verbose = False, alsave = "interceptAlphasTemp.npy"):

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
				if(kalman):
					bestAlphas[self.B.index(b)] = self.Gamma[np.argmax([self.continuousDot(self.Gamma[j],b) for j in range(0,len(self.Gamma))])];
					Value[self.B.index(b)] = self.continuousDot(bestAlphas[self.B.index(b)],b); 
				else:
					
					x = int(round(b[0]*10)); 
					bestAlphas[self.B.index(b)] = self.Gamma[x][np.argmax([self.mixedDot(self.Gamma[x][j],b) for j in range(0,len(self.Gamma[x]))])];
					Value[self.B.index(b)] = self.mixedDot(bestAlphas[self.B.index(b)],b); 

			GammaNew = [0]*51; 
			for i in range(0,51):
				GammaNew[i] = []; 


			BTilde = copy.deepcopy(self.B); 
			 
		
			while(len(BTilde) > 0):
				 
				#print(len(BTilde)); 

				if(self.exitFlag):
					break; 

				b = random.choice(BTilde); 


				BTilde.remove(b); 

				if(kalman):
					al = self.kalmanBackup(b); 
				else:
					self.newMixedPreCalc(b); 
					al = self.mixedBackup(b); 

				if(kalman):
					if(self.continuousDot(al,b) < Value[self.findB(b)]):
						index = 0; 
						for h in self.B:
							if(b.comp(h)):
								index = self.B.index(h); 
						al = bestAlphas[index]; 
				else:
					if(self.mixedDot(al,b) < Value[self.findMixedB(b)]):
						index = 0; 
						for h in self.B:
							if(b[1].comp(h[1]) and b[0] == h[0]):
								index = self.B.index(h); 
						al = bestAlphas[index]; 



				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
					if(kalman):
						if(self.continuousDot(al,bprime) >= Value[self.findB(bprime)]):
							BTilde.remove(bprime); 
					else:
						if(bprime[0] == b[0] and self.mixedDot(al,bprime) >= Value[self.findMixedB(bprime)]):
							BTilde.remove(bprime); 
							GammaNew[int(round(bprime[0]*10))] += [al]; 

			'''
			#clean any location that have no alphas
			for i in range(0,len(GammaNew)):
				if(GammaNew[i] == [] or len(GammaNew[i]) == 0):
					GammaNew[i] = self.Gamma[i]; 
			'''
			 
			'''
			for i in range(0,len(GammaNew)):
				for j in range(0,len(GammaNew[i])):
					print(GammaNew[i][j].action); 
			'''


			if(verbose and self.exitFlag == False and kalman):
				print("Number of Alphas: " + str(len(GammaNew))); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Average number of mixands: " + str(av)); 
			elif(verbose and self.exitFlag == False):
				suma = 0; 
				for i in range(0,len(GammaNew)):
					suma += len(GammaNew[i]); 
				print("Average Number of Alphas: " + str((suma/len(GammaNew)))); 
				av = 0.0; 
				for i in range(0,len(GammaNew)):
					tmpav = 0.0; 
					for j in range(0,len(GammaNew[i])):
						tmpav += GammaNew[i][j].size; 
					tmpav = tmpav/len(GammaNew[i]); 
					av += tmpav; 
				av = av/len(GammaNew);  
				print("Average number of mixands: " + str(av)); 

			if(self.exitFlag == False and kalman):
				if(counter < N-1):
					for i in range(0,len(GammaNew)):
						GammaNew[i].condense(max_num_mixands=maxMix);
				elif(counter == N-1):
					for i in range(0,len(GammaNew)):
						GammaNew[i].condense(max_num_mixands=finalMix);
			elif(self.exitFlag == False):
				if(counter < N-1):
					for i in range(0,len(GammaNew)):
						for j in range(0,len(GammaNew[i])):
							if(GammaNew[i][j].size > 200):
								GammaNew[i][j] = GM(); 
								GammaNew[i][j].addG(Gaussian(2.5,200,0.05));
								GammaNew[i][j].action = 2;  
							GammaNew[i][j].condense(max_num_mixands=maxMix);
				elif(counter == N-1):
					for i in range(0,len(GammaNew)):
						for j in range(0,len(GammaNew[i])):
							GammaNew[i][j].condense(max_num_mixands=finalMix);

			if(verbose and self.exitFlag == False and kalman):
				#GammaNew[0].display(); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Reduced number of mixands: " + str(av)); 
				print("Actions: " + str([GammaNew[i].action for i in range(0,len(GammaNew))])); 
				print("");
			elif(verbose and self.exitFlag == False):
				#GammaNew[0].display(); 
				av = 0.0; 
				for i in range(0,len(GammaNew)):
					tmpav = 0.0; 
					for j in range(0,len(GammaNew[i])):
						tmpav += GammaNew[i][j].size; 
					tmpav = tmpav/len(GammaNew[i]); 
					av += tmpav; 
				av = av/len(GammaNew);
				print("Reduced number of mixands: " + str(av)); 
				print("Actions: " + str([[GammaNew[i][j].action for j in range(0,len(GammaNew[i]))] for i in range(0,len(GammaNew))])); 
				print("");

			if(self.exitFlag == False):
				f = open(alsave,"w"); 
				np.save(f,self.Gamma); 
				f.close(); 
				self.Gamma = copy.deepcopy(GammaNew); 

		f = open(alsave,"w"); 
		np.save(f,self.Gamma); 
		f.close(); 
				

	

	def mixedPreCalc(self,x):
		G = self.Gamma; 
		pz = self.pz; 


		als = [[[0 for i in range(0,len(pz))] for j in range(0,len(self.delA))] for k in range(0,len(G[x]))]; 

		for j in range(0,len(G[x])):
			for a in range(0,len(self.delA)):
				for o in range(0,len(pz)):
					als[j][a][o] = GM(); 
					for k in range(0,G[x][j].size):
						for l in range(0,pz[x][o].size): 
							k1 = G[x][j].Gs[k]; 
							l1 = pz[o].Gs[l];  

							we1 = k1.weight*l1.weight;

							#The k1.mean is a list for some reason? 
							if(isinstance(k1.mean,list)):
								k1.mean = k1.mean[0]; 
							#print(k1.mean,l1.mean,self.delA[a]); 



							we2 = mvn.pdf(x,k1.mean - l1.mean - self.delA[a], l1.var + k1.var + self.delAVar[a]); 
							c2 = (np.matrix(l1.var).I + np.matrix(k1.var).I).I; 
							c1 = c2*(np.matrix(l1.var).I*np.matrix(l1.mean) + np.matrix(k1.var).I*np.matrix(k1.mean)); 
							#we3 = mvn.pdf(y + self.ydel(y),c1,c2 + self.delAVar); 
							w = we1*we2; 
							mean = c1-self.ydel(); 
							var = c2 + self.delAVar[a]; 



							als[j][a][o].addG(Gaussian(mean,var,w)); 
		
	

		self.preAls = als; 


	#Experimental, based on sarsop update methods
	def newMixedPreCalc(self,b):
		G = self.Gamma; 
		pz = self.pz; 

		numAls = 0; 
		for i in range(0,len(G)):
			for j in range(0,len(G[i])):
				numAls += 1; 

		als = [[[0 for i in range(0,len(pz))] for a in range(0,len(self.delA))] for x in range(0,numAls)]; 

		count = -1; 
		for x in range(0,len(G)):
			for j in range(0,len(G[x])):
				count += 1; 
				for a in range(0,len(self.delA)):
					for o in range(0,len(pz)):
						
						bnew = [0,0]; 
						

						bnew[0] = np.random.normal(b[0]+self.delA[a],self.delAVar[a],1); 
						bnew[1] = self.beliefUpdate(a,o,b); 
						

						als[count][a][o] = G[x][np.argmax([self.mixedDot(G[x][j],bnew) for j in range(0,len(G[x]))])]; 

		

		self.preAls = als; 


	def mixedBackup(self,b):
		x = int(round(b[0])); 
		c = b[1]; 
		R = self.r[x*10]; 
		pz = self.pz; 



		als = self.preAls; 

		bestVal = -10000000000; 
		bestAct= 0; 
		bestGM = []; 

		for a in range(0,len(self.delA)):
			suma = GM(); 
			for o in range(0,len(pz)):
				suma.addGM(als[np.argmax([self.mixedDot(als[j][a][o],b) for j in range(0,len(als))])][a][o]); 
			suma.scalerMultiply(self.discount); 
			
			suma.addGM(R); 

			tmp = self.mixedDot(suma,b);
			#print(a,tmp); 
			if(tmp > bestVal):
				bestAct = a; 
				bestGM = suma; 
				bestVal = tmp; 
		bestGM.action = bestAct; 

		 


		return bestGM; 



	def kalmanBackup(self,b):
		G = self.Gamma; 
		R = self.r; 
		

		bestG = G.index(G[np.argmax([self.continuousDot(j,b) for j in G])]); 
		bestAct = G[bestG].action; 
		bestVal = -10000000000;
		
		#for each alpha in G
		for g in G:
			#for each action
			for a in range(0,len(self.delA)):
				#calculate the inner product
				val = self.stateInnerProduct(g,b,a); 
				#find the max
				if(val > bestVal):
					bestVal = val; 
					bestG = g; 
					bestAct = a; 

		bestGM = copy.deepcopy(R[bestAct]); 
		
		bestGM.action = bestAct; 
		for g in bestG.Gs:
			w = 0.95*g.weight; 
			chi = (self.Qt + self.Ct*np.matrix(g.var)*self.Ct.T).I; 
			c2 = (self.Qt + self.Ct*np.matrix(g.var)*self.Ct.T).I * np.matrix(g.var)*self.Qt; 
			
			m = np.matrix(g.mean).T - self.Bt*np.matrix(self.delA[a]).T + chi*np.matrix(g.var)*self.Ct*np.matrix(g.mean).T;# - self.delA[2].T; 
			v = self.Qt*chi*chi.I*(self.Qt*chi).T + c2 +  self.Rt;


			bestGM.addG(Gaussian(m.T.tolist()[0],v.tolist(),w)); 

		
		return bestGM;  


	def initializeKalman(self):

		self.At = np.matrix(1); 
		self.Bt = np.matrix(1);  
		self.Ct = np.matrix(1);  
		self.Rt = np.matrix(5);  
		self.Qt = np.matrix(5); 

	def initializeMixed(self):
		
		'''
		self.pz = [0]*51; 

		for i in range(0,51):
			self.pz[i] = [GM(),GM()];
			self.pz[i][0].addG(Gaussian(i/10,1/16,0.5)); 

			for j in [x/10 for x in range(i-5,-5,-5)]:
				self.pz[i][1].addG(Gaussian(j,1/16,0.5)); 

			for j in [x/10 for x in range(i+5,55,5)]:
				self.pz[i][1].addG(Gaussian(j,1/16,0.5)); 
		'''

		self.pz = [GM(),GM()]; 
		self.pz[0].addG(Gaussian(0,1/16,0.5)); 
		for j in [x/2.0 for x in range(-12,12)]:
				if(abs(j) > .5):
					self.pz[1].addG(Gaussian(j,1/16,0.5)); 

		
		
		

	def buildTransition(self,kalman,order = 0):
		if(kalman):
			self.delA = [-1,1,0];  
		else:
			self.delA = [-.5,-.5,0]; 
			self.delAVar = [.25,.25,0.0001]; 
		
	def ydel(self):
		return 0; 
		

	def buildReward(self,kalman,order = 0):

		if(kalman):
			self.r = [0]*3;
			self.r[0] = GM(); 
			
			self.r[1] = GM();
			self.r[0].addG(Gaussian(0,1000,-1)); 
			self.r[1].addG(Gaussian(0,1000,-1)); 


			self.r[2] = GM(); 
			self.r[2].addG(Gaussian(25,12.5,-40)); 
			self.r[2].addG(Gaussian(-25,12.5,-40));
			self.r[2].addG(Gaussian(-3,0.2,2)); 
		else:
			
			#reward for catch is 3 apparently...

			self.r = [0]*51; 
			for i in range(0,51):
				self.r[i] = GM(); 
				self.r[i].addG(Gaussian(i/10,1/16,1.9)); 

				for j in [x/2.0 for x in range(-1,12)]:
					if(abs(j-i/10) > .5):
						self.r[i].addG(Gaussian(j,1/16,-0.5)); 

	 



		 




				

				



	def continuousDot(self,a,b):
		suma = 0; 

		if(isinstance(a,np.ndarray)):
			a = a.tolist(); 
			a = a[0]; 

		if(isinstance(a,list)):
			a = a[0];  

		for k in range(0,a.size):
			for l in range(0,b.size):
				
				
				suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean,(np.matrix(a.Gs[k].var) +np.matrix(b.Gs[l].var)).tolist()); 

		return suma; 

	def mixedDot(self,a,b):
		suma = 0; 
		c = b[1]; 
		for k in range(0,a.size):
			for l in range(0,c.size): 

				suma += a.Gs[k].weight*c.Gs[l].weight*mvn.pdf(c.Gs[l].mean,a.Gs[k].mean,(np.matrix(a.Gs[k].var) +np.matrix(c.Gs[l].var)).tolist()); 

		return suma; 


	def findB(self,b):
		for beta in self.B:
			if(beta.comp(b)):
				return self.B.index(beta); 

	def findMixedB(self,b):
		for beta in self.B:
			if(beta[0] == b[0] and beta[1].comp(b[1])):
				return self.B.index(beta); 

	def stateInnerProduct(self,g,b,act):
		suma = 0; 

		for k in range(0,g.size):
			for h in range(0,b.size):
				c = g.Gs[k]; 
				d = b.Gs[h]; 

				c2 = (self.Qt + self.Ct*np.matrix(c.var)*self.Ct.T).I * np.matrix(c.var)*self.Qt; 
				chi = (self.Qt + self.Ct*np.matrix(c.var)*self.Ct.T).I; 

				

				m = np.matrix(d.mean).T + self.Bt*np.matrix(self.delA[act]).T - chi*np.matrix(c.var)*self.Ct*np.matrix(c.mean).T; 
				v = self.Qt*chi*(self.Ct*np.matrix(c.var)*self.Ct.T + self.Qt)*(self.Qt*chi).T + self.At*np.matrix(d.var)*self.At.T + c2 +  self.Rt;

				

				suma += c.weight*d.weight * mvn.pdf(c.mean,mean = m.T.tolist()[0], cov = v); 

		return suma; 


	def kalmanFilter(self,a,z,B):
		#predict
		xhat = self.At*B.Gs[0].mean + self.Bt*self.delA[a]; 
		phat = self.At*B.Gs[0].var*self.At.T + self.Rt; 

		#update
		K = phat*self.Ct.T*(self.Ct*phat*self.Ct.T + self.Qt).I; 
		x = xhat + K*(z-self.Ct*xhat); 
		P = (1 - K*self.Ct)*phat; 

		x = x.tolist()[0][0]; 
		P = P.tolist()[0][0]; 

		b = GM(); 
		b.addG(Gaussian(x,P,1)); 
		return b; 


	def beliefUpdate(self,a,o,b):

		btmp = GM(); 
		
		for i in self.pz[o].Gs:
			for j in b[1].Gs:
				

				tmp = mvn.pdf(j.mean,i.mean,np.matrix(i.var) + np.matrix(j.var) + np.matrix(self.delAVar[a])); 
				#print(i.weight,j.weight,tmp); 
				w = i.weight*j.weight*tmp.tolist(); 

				sig = (np.matrix(i.var).I + (np.matrix(j.var)+ np.matrix(self.delAVar[a])).I).I.tolist(); 

				#sstmp = np.matrix(i.var).I*np.transpose(i.mean) + np.matrix(self.covAdd(j.var + self.delAVar)).I*np.transpose(np.add(np.matrix(j.mean),np.matrix(delA[a])));
				sstmp1 = np.matrix(i.var).I*np.transpose(np.matrix(i.mean)); 
				sstmp2 = (np.matrix(j.var) + np.matrix(self.delAVar[a])).I; 
				sstmp21 = np.add(np.matrix(j.mean),np.matrix(self.delA[a]));
				

				sstmp3 = sstmp1 + sstmp2*np.transpose(sstmp21);  
				smean = np.transpose(sig*sstmp3).tolist()[0]; 

				btmp.addG(Gaussian(smean,sig,w)); 


		#print(btmp.size); 
		#btmp.condense(self.nB); 
		btmp.normalizeWeights();
		#btmp.display();  

		return btmp; 



	def simBeliefs(self,num,fileName = "interceptBeliefs1.npy"):
		B = [0]*(num + 40); 
		B[0] = GM(); 
		B[0].addG(Gaussian(0,100,1)); 
		xs = [0]; 

		x = 0; 

		for i in range(0,num-1):
			b = B[i]; 
			a = random.choice([0,1,2]); 
			if(a == 0):
				x = np.random.normal(x-1,self.Rt,1); 
			elif(a == 1):
				x = np.random.normal(x+1,self.Rt,1);
			if(x > 20):
				x = 20; 
			elif(x < -20):
				x = -20; 
			z = np.random.normal(x,self.Qt,1); 

			B[i+1] = self.kalmanFilter(a,z,b); 

			#print(z); 
			xs += [x];  

		print(max(xs), min(xs), sum(xs) / float(len(xs))); 

		for i in range(num,num+40):
			B[i] = GM(); 
			B[i].addG(Gaussian(i-num-20,1,1)); 

		f = open(fileName,"w"); 
		np.save(f,B); 


	def simBelMixed(self,num,numMix,fileName = "remixedBeliefs1.npy"):
		
		B = [0]*51*num; 
		
		#i is x position

		for i in range(0,51):
			for j in range(0,num):
				B[i*num + j] = [i/10,GM()]; 
				for k in range(0,numMix):
					B[i*num + j][1].addG(Gaussian(random.random()*5,0.25,1/numMix)); 
				
			 


		while 0 in B: B.remove(0); 

		
		f = open(fileName,"w"); 
		np.save(f,B); 
 


	def simulate(self,start = -10,length = 40,alphaFile = "interceptAlphas1.npy"):
		Gamma = np.load(alphaFile);
		B = GM(); 
		B.addG(Gaussian(0,100,1)); 

		bels = [B]; 

		x = start; 
		a = 0; 
		count = 0; 


		while(not (x==3 and a == 2) and count < length):
			count += 1; 
			a = Gamma[np.argmax([self.continuousDot(j,B) for j in Gamma])].action; 
			if(a == 0):
				x = np.random.normal(x-1,self.Rt,1); 
			elif(a == 1):
				x = np.random.normal(x+1,self.Rt,1);
			if(x > 20):
				x = 20; 
			elif(x < -20):
				x = -20; 
			z = np.random.normal(x,self.Qt,1); 
			B = self.kalmanFilter(a,z,B);
			B.action = x;  
			bels = bels + [B]; 
			#print(x,a,B.Gs[0].mean,B.Gs[0].var); 
			 
		return bels; 

	def mixSimulate(self,start = [0,4],length = 40,alphaFile = "mixedAlphas1.npy"):
		Gamma = np.load(alphaFile);
		B = [0,GM()]; 
		B[1].addG(Gaussian(0,100,1)); 

		bels = [B]; 

		[x,y] = start; 
		a = 0; 
		count = 0;
		while(abs(x-y) > 1 and count < length):
			count += 1; 

			a = Gamma[int(round(B[0])) + 20][np.argmax([self.mixedDot(j,B) for j in Gamma[int(round(B[0])) + 20] ])].action; 
			

			#TODO: Minus sign in the random
			if(a == 0 or a == 1):
				x = np.random.normal(x+self.delA[a],self.delAVar[a],1); 
			elif(a == 2 or a == 3):
				x = x; 
			y = y + self.ydel(); 
			if(x > 20):
				x = 20; 
			elif(x < -20):
				x = -20; 
			if(y > 20):
				y = 20; 
			elif(y < -20):
				y = -20; 

			print(x,y,a); 

			z = 0; 
			
			if(y >= -10 and y <10):
				z = 2; 
			elif(y >= 10):
				z = 1; 
			
			#z = int(round(y))+20;   

			tmp = self.beliefUpdate(a,z,B[1])
			tmp.action = y; 
			B = [x,tmp]; 

			bels = bels + [B]; 

		return bels; 



	
	def ani(self,numFrames = 20):
		fig, ax = plt.subplots()

		a = np.linspace(-20,20,num = 1000); 

		for t in range(0,numFrames):
		 	if t == 0:
				points, = ax.plot(a,self.bels[t].plot(vis = False)); 
				single = ax.scatter(self.bels[t].action,0.5); 
				ax.set_xlim(-20, 20) 
				ax.set_ylim(0, 5)
			else: 
				ax.cla(); 
				points, = ax.plot(a,self.bels[t].plot(vis = False))
				single = ax.scatter(self.bels[t].action,0.5,linewidths = 4); 
				ax.set_xlim(-20, 20) 
				ax.set_ylim(0, 5)
		    	

			plt.pause(0.5)
	
	def aniMixed(self,numFrames = 60):
		fig, ax = plt.subplots()

		a = np.linspace(0,5,num = 1000); 

		for t in range(0,numFrames):
		 	if t == 0:
				points, = ax.plot(a,self.bels[t][1].plot(vis = False)); 
				single = ax.scatter(self.bels[t][0],0.5); 
				single2 = ax.scatter(self.bels[t][1].action,1); 
				ax.set_xlim(0, 5) 
				ax.set_ylim(0, 4)
			else: 
				ax.cla(); 
				points, = ax.plot(a,self.bels[t][1].plot(vis = False))
				single = ax.scatter(self.bels[t][0],0.5,linewidths = 4, color = "Blue"); 
				single2 = ax.scatter(self.bels[t][1].action,1,linewidths = 4, color = "Red"); 
				ax.set_xlim(0, 5) 
				ax.set_ylim(0, 4)
		    	

			plt.pause(0.5)


if __name__ == "__main__":



	alload = "remixedAlphas1.npy"; 
	alsave = "remixedAlphas1.npy"; 
	belload = "remixedBeliefs1.npy"; 
	sol = True;  
	sim = False;   
	plo = False;  
	simbel = False;
	kal = False; 


	a = InterceptPolicyGenerator(kalman = False,beliefFile = belload); 
	
	'''
	Gamma = np.load(alload);
	fig2 = plt.figure();
	spot = 50; 
	for i in range(0,len(Gamma[spot])):
		ax2 = fig2.add_subplot(len(Gamma[spot]),1,i+1);
		c = Gamma[spot][i].plot(0,5,vis=False); 
		ax2.plot(c,linewidth = 3); 
		print(Gamma[spot][i].action); 

	plt.show();
	'''
	
	



	if(sol):
		a.solve(N = 100, maxMix = 100, finalMix = 100, kalman = kal, verbose = True, alsave = alsave);


	if(sim and kal):
		a.bels = a.simulate(start = 15, length = 40,alphaFile= alload); 
		a.ani(numFrames = 40); 
	elif(sim):
		a.bels = a.mixSimulate(start = [0,4], length = 80, alphaFile = alload); 
		a.aniMixed(numFrames = 80); 

	if(simbel and kal):
		a.simBeliefs(num=500); 
	elif(simbel):
		a.simBelMixed(num = 2,numMix = 2); 



	 



	'''
	Alphas Generations:
	1: Good, but too broad on the 2s 
	2: A little better, same problem
	3: 1s on the wrong side? 
	4: Too broad still...
	5: Just fine
	6: Higher uncertainty, worked out alright
	7: Pretty great (actually meh)
	8:---
	9:meh
	10:
	11: Good
	12: GOOD
	13:
	14: GOOOOOD
	15: meh
	16:GOOOOOOOOD


	Mixed:
	3: Half decent
	5: Works, for all starting!!!!, didn't have rewards for left and right
	6: Better than 5, working start points: (-20,0),(20,0)
	7:

	Remixed:
	1:

	'''


	
	if(plo and not kal):

		acts = [-1]*41; 

		als = np.load(alload); 
		if(isinstance(als,np.ndarray)):
			als = als.tolist();
			

		acts = [-1]*41; 
		for i in range(-20,21):
			b = [i,GM()]; 
			b[1].addG(Gaussian(i,1,1)); 
			acts[i+20] = als[i+20][np.argmax([a.continuousDot(j,b[1]) for j in als[i+20]])].action;
		print(acts); 



		fig2 = plt.figure();
		fig2.suptitle("Alpha Functions corresponding to each action"); 
		
		c = np.linspace(-20,20,num = 1000);
		b0 = [0.0]*1000; 
		b1 = [0.0]*1000; 
		b2 = [0.0]*1000; 
		for x in range(0,41):
			for ac in als[x]:
				if(ac.action == 0):
					b0 += ac.plot(vis = False); 
				elif(ac.action == 1):
					b1 += ac.plot(vis = False); 
				else:
					b2 += ac.plot(vis = False); 
		

		ax2 = fig2.add_subplot(3,1,1);
		ax2.plot(c,b0,linewidth = 3); 
		ax2.set_title("Action: Move Left"); 
		ax3 = fig2.add_subplot(3,1,2); 
		ax3.plot(c,b1,linewidth = 3); 
		ax3.set_title("Action: Move Right")
		ax4 = fig2.add_subplot(3,1,3); 
		ax4.set_title("Action: Declare Victory"); 
		ax4.plot(c,b2,linewidth = 3);


		fig4 = plt.figure(); 
		testpoint = 21; 
		for i in range(0,len(als[testpoint])):
			ax = fig4.add_subplot(len(als[testpoint]),1,i); 
			ax.plot(c,als[testpoint][i].plot(vis=False)); 
			print(als[testpoint][i].action); 


		plt.show(); 




	if(plo and kal):
		
		als = np.load(alload); 

		acts = [-1]*40; 
		for i in range(-20,20):
			b = GM(); 
			b.addG(Gaussian(i,1,1)); 
			act = als[np.argmax([a.continuousDot(j,b) for j in als])].action;
			acts[i+20] = act; 
		print(acts); 


		fig2 = plt.figure();
		fig2.suptitle("Alpha Functions corresponding to each action"); 
		
		c = np.linspace(-20,20,num = 1000);
		b0 = [0.0]*1000; 
		b1 = [0.0]*1000; 
		b2 = [0.0]*1000; 
		for ac in als:
			if(ac.action == 0):
				b0 += ac.plot(vis = False); 
			elif(ac.action == 1):
				b1 += ac.plot(vis = False); 
			else:
				b2 += ac.plot(vis = False); 
		ax2 = fig2.add_subplot(3,1,1);
		ax2.plot(c,b0,linewidth = 3); 
		ax2.set_title("Action: Move Right"); 
		ax3 = fig2.add_subplot(3,1,2); 
		ax3.plot(c,b1,linewidth = 3); 
		ax3.set_title("Action: Move Left")
		ax4 = fig2.add_subplot(3,1,3); 
		ax4.set_title("Action: Declare Victory"); 
		ax4.plot(c,b2,linewidth = 3);



		fig2.text(0.5,0.04,"Position",ha = 'center',va='center'); 
		fig2.text(0.06,0.5,"Value",ha = 'center',va='center',rotation = 'vertical'); 


		fig3 = plt.figure(); 
		ax1 = fig3.add_subplot(1,1,1); 
		ax1.plot(c,b0-b1,linewidth = 3); 
		ax1.axhline(y = 0)
		

		fig4 = plt.figure(); 
		for i in range(0,len(als)):
			ax = fig4.add_subplot(len(als),1,i); 
			ax.plot(c,als[i].plot(vis=False)); 




		plt.show(); 




	











	
	


	