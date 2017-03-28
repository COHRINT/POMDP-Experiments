from __future__ import division
'''
****************************************************
File: InterceptPolicyGenerator.py
Written By: Luke Burks
October 2016

This is just a simplified copy of the Intercept Policy 
Generator for a full regular pomdp case.

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
import time
import matplotlib.image as mgimg

#TODO: Include stop-flag criteria


class InterceptTestGenerator:

	def __init__(self,beliefFile = None,dep = False,depFac = 0.1):
		
		 

		#Initialize exit flag
		self.exitFlag = False; 

		self.buildTransition(); 
		self.buildObs(); 
		self.buildReward(); 
		self.discount = 0.9; 
		self.discountDeprecate = dep; 
		self.deprecateFactor = depFac; 

		if(beliefFile == None):
			self.B = [0]*5; 
			self.B[0] = GM(); 
			var = np.matrix([[1,0],[0,1]]); 
			self.B[0].addG(Gaussian([2.5,2.5],var,1)); 

			self.B[1] = GM(); 
			self.B[1].addG(Gaussian([1,5],var,1));


			self.B[2] = GM(); 
			self.B[2].addG(Gaussian([5,1],var,1));

			self.B[3] = GM(); 
			self.B[3].addG(Gaussian([0,0],var,1));

			self.B[4] = GM(); 
			self.B[4].addG(Gaussian([5,5],var,1));



			for i in range(0,100):
				tmp = GM(); 
				tmp.addG(Gaussian([random.random()*5,random.random()*5],var,1)); 
				self.B.append(tmp); 


		else:
			self.B = np.load(beliefFile).tolist(); 
			
		



		#Initialize Gamma
		self.Gamma = [copy.deepcopy(self.r)]; 
		#self.Gamma = [copy.deepcopy(self.r),copy.deepcopy(self.r),copy.deepcopy(self.r)];

		

		'''
		for i in range(0,3):
			self.Gamma[i].addG(Gaussian([0,0],[[100,0],[0,100]],-5)); 
			self.Gamma[i].action = i; 
		'''

		#TODO: This stuff....
		for i in range(0,len(self.Gamma)):
			for j in range(0,len(self.Gamma[i].Gs)):
				self.Gamma[i].Gs[j].weight = -100000; 
				#tmp = 0; 
		
		

	
		 



	def solve(self,N,maxMix = 20, finalMix = 50, verbose = False, alsave = "interceptAlphasTemp.npy"):

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
			 

			#self.preComputeAls(); 
			self.newPreComputeAls(); 


			while(len(BTilde) > 0):

				if(self.exitFlag):
					break; 

				b = random.choice(BTilde); 

				BTilde.remove(b); 

				al = self.backup(b); 

			
				#TODO: You added the else here
				if(self.continuousDot(al,b) < Value[self.findB(b)]):
					index = 0; 
					for h in self.B:
						if(b.comp(h)):
							index = self.B.index(h); 
					al = bestAlphas[index]; 
				else:
					index = 0; 
					for h in self.B:
						if(b.comp(h)):
							index = self.B.index(h);
					bestAlphas[index] = al; 

				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
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
						#TODO: Switch back to kmeans
						GammaNew[i].condense(max_num_mixands=maxMix);
						#GammaNew[i] = GammaNew[i].kmeansCondensation(k = maxMix); 
				elif(counter == N-1):
					for i in range(0,len(GammaNew)):
						GammaNew[i].condense(max_num_mixands=finalMix);
						#GammaNew[i] = GammaNew[i].kmeansCondensation(k = finalMix); 

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
				f = open(alsave,"w"); 
				np.save(f,self.Gamma); 
				f.close(); 
				self.Gamma = copy.deepcopy(GammaNew); 

			'''
			if((counter+1)%5 == 0):
				for i in range(0,len(self.Gamma)):
					fig1 = plt.figure();
					print(self.Gamma[i].action); 
					self.Gamma[i].plot2D();
				for j in range(0,3):
					print(self.getAction(self.B[j])); 
			'''


		f = open(alsave,"w"); 
		np.save(f,self.Gamma); 
		f.close(); 



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

							weight = G[j].Gs[k].weight*pz[o].Gs[l].weight*mvn.pdf(pz[o].Gs[l].mean,G[j].Gs[k].mean,(np.matrix(G[j].Gs[k].var)+np.matrix(pz[o].Gs[l].var)).tolist()); 

							#get sig and ss
							sigtmp = (np.matrix(G[j].Gs[k].var).I + np.matrix(pz[o].Gs[l].var)).tolist(); 
							sig = np.matrix(sigtmp).I.tolist(); 
						
							sstmp = np.matrix(G[j].Gs[k].var).I*np.transpose(np.matrix(G[j].Gs[k].mean)) + np.matrix(pz[o].Gs[l].var).I*np.transpose(np.matrix(pz[o].Gs[l].mean)); 
							ss = np.dot(sig,sstmp).tolist(); 


							smean = (np.transpose(np.matrix(ss)) + np.matrix(self.delA[a])).tolist(); 
							sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist(); 
			
								
							als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight)); 
		self.preAls = als1; 


	#based on the idea that 1-detect = not detect
	#only to be used for binary observations
	def newPreComputeAls(self):
		G = self.Gamma; 
		R = self.r; 
		pz = self.pz; 

		als1 = [[[0 for i in range(0,2)] for j in range(0,len(self.delA))] for k in range(0,len(G))]; 

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				o = 0; 
				als1[j][a][o] = GM(); 
				for k in range(0,G[j].size):
					for l in range(0,pz[o].size): 
						#get weights wk,wl, and del
						weight = G[j].Gs[k].weight*pz[o].Gs[l].weight*mvn.pdf(pz[o].Gs[l].mean,G[j].Gs[k].mean, self.covAdd(G[j].Gs[k].var,pz[o].Gs[l].var)); 

						#get sig and ss
						sig= (np.matrix(G[j].Gs[k].var).I + np.matrix(pz[o].Gs[l].var).I).I.tolist();
						
					
						sstmp = np.matrix(G[j].Gs[k].var).I*np.transpose(np.matrix(G[j].Gs[k].mean)) + np.matrix(pz[o].Gs[l].var).I*np.transpose(np.matrix(pz[o].Gs[l].mean)); 
						ss = np.dot(sig,sstmp);


						smean = (np.transpose(np.matrix(ss)) + np.matrix(self.delA[a])).tolist();
						sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist();
		
							
						als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight)); 

				als1[j][a][1] = GM(); 
				o = 1; 
				
				for k in range(0,G[j].size):
					kap = G[j].Gs[k];
					mean = (np.matrix(kap.mean) - np.matrix(self.delA[a])).tolist();
					var = (np.matrix(kap.var) + np.matrix(self.delAVar)).tolist();
					als1[j][a][o].addG(Gaussian(mean,var,kap.weight)); 

					for l in range(0,pz[0].size):

						op = pz[0].Gs[l]; 
						var = (np.matrix(kap.var) + np.matrix(op.var)).tolist();
						weight = kap.weight*op.weight*mvn.pdf(kap.mean,op.mean,var); 


						

						c2 = (np.matrix(kap.var).I + np.matrix(op.var).I).I; 
						c1 = c2*(np.matrix(kap.var).I*np.transpose(np.matrix(kap.mean)) + np.matrix(op.var).I*np.transpose(np.matrix(op.mean)))

						me = np.transpose((c1 - np.transpose(np.matrix(self.delA[a])))).tolist()[0]; 

					

						als1[j][a][o].addG(Gaussian(me,(c2+np.matrix(self.delAVar)).tolist(),-weight)); 
						
						 



		self.preAls = als1; 


	def backup(self,b):
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
			if(self.discountDeprecate):
				self.discount = self.discount - self.discount*self.deprecateFactor; 
			suma.addGM(R); 

			tmp = self.continuousDot(suma,b);
			#print(a,tmp); 
			if(tmp > bestVal):
				bestAct = a; 
				bestGM = suma; 
				bestVal = tmp; 

		bestGM.action = bestAct; 

		return bestGM;  



	def getAction(self,b):
		act = self.Gamma[np.argmax([self.continuousDot(j,b) for j in self.Gamma])].action;
		return act; 

	def getGreedyAction(self,b,x):
		MAP = b.findMAP2D()[1];
		copLoc = x[0]; 
		if(MAP > copLoc+0.5):
			return 1; 
		elif(MAP < copLoc-0.5):
			return 0; 
		else:
			return 2; 

	def MDPValueIteration(self):
		#Intialize Value function
		self.ValueFunc = copy.deepcopy(self.r); 
		for g in self.ValueFunc.Gs:
			g.weight = -1000; 

		comparision = GM(); 
		comparision.addG(Gaussian([0,0],[[1,0],[0,1]],1)); 

		uniform = GM(); 
		for i in range(0,10):
			for j in range(0,10):
				uniform.addG(Gaussian([i/2,j/2],[[1,0],[0,1]],1)); 

		count = 0; 

		#until convergence
		while(not self.ValueFunc.comp(comparision) and count < 100):
			comparision = copy.deepcopy(self.ValueFunc); 
			count += 1;
			#print(count); 
			maxVal = -10000000; 
			maxGM = GM(); 
			for a in range(0,2):
				suma = GM(); 
				for g in self.ValueFunc.Gs:
					mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist(); 
					var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
					suma.addG(Gaussian(mean,var,g.weight));  
				suma.addGM(self.r); 
				tmpVal = self.continuousDot(uniform,suma); 
				if(tmpVal > maxVal):
					maxVal = tmpVal; 
					maxGM = copy.deepcopy(suma); 

			maxGM.scalerMultiply(self.discount); 
			maxGM.condense(20); 
			self.ValueFunc = copy.deepcopy(maxGM); 

		#self.ValueFunc.display(); 
		#self.ValueFunc.plot2D(); 
		print("MDP Value Iteration Complete"); 


	def getMDPAction(self,x):
		maxVal = -10000000; 
		maxGM = GM();
		bestAct = 0;  
		for a in range(0,2):
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
			if(beta.comp(b)):
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

				suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean, self.covAdd(a.Gs[k].var,b.Gs[l].var)); 
		return suma; 

	#TODO: You changed the variance for the cop
	#TODO: You changed the length of the transitions

	#movement variance is 0.25 for the robber, stationary is 0.0001
	def buildTransition(self):
		self.delAVar = [[0.0001,0],[0,0.25]]; 
		self.delA = [[-0.5,0],[0.5,0],[0,0]]; 


	def buildObs(self):
		self.pz = [GM(),GM()];  
		var = [[1,.7],[.7,1]]; 
		for i in range(-1,7):
			self.pz[0].addG(Gaussian([i,i],var,1));

		for i in range(1,13):
			for j in range(-1,7):
				self.pz[1].addG(Gaussian([j,j+i],var,1)); 
				self.pz[1].addG(Gaussian([j,j-i],var,1));  

		self.pz[0].condense(3); 
		self.pz[1].condense(20); 



	def buildReward(self):
		self.r = GM(); 
		var = [[1,.9],[.9,1]];

		for i in range(-1,7):
			self.r.addG(Gaussian([i,i],var,5.6250));
		for i in range(1,13):
			for j in range(-1,7):
				self.r.addG(Gaussian([j,j+i],var,-1)); 
				self.r.addG(Gaussian([j,j-i],var,-1));


		self.r.condense(20); 


	def beliefUpdate(self,b,a,o,maxMix = 10):

		btmp = GM(); 

		for i in self.pz[o].Gs:
			for j in b.Gs:
				
				tmp = mvn.pdf(np.add(np.matrix(j.mean),np.matrix(self.delA[a])).tolist(),i.mean,self.covAdd(self.covAdd(i.var,j.var),self.delAVar))  
				#print(i.weight,j.weight,tmp); 
				w = i.weight*j.weight*tmp.tolist(); 

				sig = (np.add(np.matrix(i.var).I, np.matrix(self.covAdd(j.var, self.delAVar)).I)).I.tolist(); 

				#sstmp = np.matrix(i.var).I*np.transpose(i.mean) + np.matrix(self.covAdd(j.var + self.delAVar)).I*np.transpose(np.add(np.matrix(j.mean),np.matrix(delA[a])));
				sstmp1 = np.matrix(i.var).I*np.transpose(np.matrix(i.mean)); 
				sstmp2 = np.matrix(self.covAdd(j.var,self.delAVar)).I; 
				sstmp21 = np.add(np.matrix(j.mean),np.matrix(self.delA[a]));
				

				sstmp3 = sstmp1 + sstmp2*np.transpose(sstmp21);  
				smean = np.transpose(sig*sstmp3).tolist()[0]; 

				btmp.addG(Gaussian(smean,sig,w)); 

		
		btmp.condense(maxMix); 
		btmp.normalizeWeights();

		return btmp; 


	def simulate(self,policy = "interceptAlphasTemp.npy",initialPose = [1,5],initialBelief = None, numSteps = 20,mul = False,MDP = False,human = False,greedy = False,randSim = False,belSave = 'tmpbelSave.npy',beliefMaxMix = 10,verbose = True):

		if(initialBelief == None):
			b = GM(); 
			b.addG(Gaussian([initialPose[0],2.5],[[0.01,0],[0,4]])); 
		else:
			b = initialBelief; 

		if(human):
			fig,ax = plt.subplots(); 
		elif(MDP and mul == False):
			self.MDPValueIteration(); 

		x = initialPose; 
		allX = []; 
		allX.append(x); 
		allX0 = [];
		allX0.append(x[0]);
		allX1 = []; 
		allX1.append(x[1])

		reward = 0; 
		allReward = [0]; 
		allB = []; 
		allB.append(b); 

		allAct = []; 

		if(randSim):
			for i in range(0,6):
				for j in range(0,6):
					x = [i,j];
					b = GM(); 
					b.addG(Gaussian([x[0],2.5],[[0.01,0],[0,4]])); 
					for k in range(0,numSteps):
						act = random.randint(0,2); 
						x = np.random.multivariate_normal([x[0] + self.delA[act][0],x[1] + self.delA[act][1]],self.delAVar,size =1)[0].tolist();
						
						x[0] = min(x[0],5); 
						x[0] = max(x[0],0); 
						x[1] = min(x[1],5); 
						x[1] = max(x[1],0); 
						if(abs(x[0]-x[1]) <= 0.5):
							z = 0; 
						else:
							z = 1; 

						b = self.beliefUpdate(b,act,z,beliefMaxMix); 
						allB.append(b);
						allX.append(x); 
						allX0.append(x[0]);
						allX1.append(x[1]);
			f = open(belSave,"w"); 
			np.save(f,allB); 

			#allB[numSteps].plot2D(); 
			print(max(allX0), min(allX0), sum(allX0) / float(len(allX0)));
			print(max(allX1), min(allX1), sum(allX1) / float(len(allX1)));
		else:
			self.Gamma = np.load(policy); 

			for count in range(0,numSteps):
				
				if(human):

					ax.cla(); 
					[z,y,c] = b.plot2D(vis = False);
					suma = [0]*len(c); 
					for i in range(0,len(c)):
						for j in range(0,len(c[i])):
							suma[i] += c[j][i]; 

					sumaA = sum(suma); 
					for i in range(0,len(suma)):
						suma[i] = suma[i]/sumaA; 

					ax.plot(z,suma); 
					col = 'blue'; 
					if(abs(x[0] - x[1]) <= 0.5):
						col = 'green'; 
					cop = ax.scatter(x[0],0,color = col,s = 400);
					ax.set_ylim([0,max(suma)]); 
					ax.set_xlim([0,5]); 
					plt.pause(0.5); 
					act = -1; 
					while(act == -1):
						try:
							act = int(raw_input('Action?'));
							if(act == 99):
								break; 
						except:
							print("Please enter a valid action...");


					 
				elif(greedy):
					act = self.getGreedyAction(b,x); 
				elif(MDP):
					act = self.getMDPAction(x); 
					#print(act); 
				else:
					act = self.getAction(b);

 				if((x[0] == 0 and act == 0) or (x[1] == 5 and act == 1)):
 					act = 2; 
				

				x = np.random.multivariate_normal([x[0] + self.delA[act][0],x[1] + self.delA[act][1]],self.delAVar,size =1)[0].tolist();
				
				allAct.append(act); 
				x[0] = min(x[0],5); 
				x[0] = max(x[0],0); 
				x[1] = min(x[1],5); 
				x[1] = max(x[1],0); 

				if(abs(x[0]-x[1]) <= 0.5):
					z = 0; 
				else:
					z = 1; 

				if(not MDP):
					b = self.beliefUpdate(b,act,z,beliefMaxMix); 
				allB.append(b);
				allX.append(x); 
				allX0.append(x[0]);
				allX1.append(x[1]);

				if(abs(x[0]-x[1]) <= 0.5):
					reward += 3; 
					allReward.append(reward); 
				else:
					reward -= 1; 
					allReward.append(reward); 


			
			allAct.append(-1);
			if(verbose):
				print("Simulation Complete. Accumulated Reward: " + str(reward));  
			return [allB,allX0,allX1,allAct,allReward]; 

	def plotRewardErrorBounds(self,allSimRewards):
		#find average reward 
		averageRewards = copy.deepcopy(allSimRewards[0]);
		
		for i in range(1,simCount):
			for j in range(0,len(allSimRewards[i])):
				averageRewards[j] += allSimRewards[i][j];

		for i in range(0,len(averageRewards)):
			averageRewards[i] = averageRewards[i]/len(allSimRewards); 

		#find sigma bounds
		sampleVariances = [0 for i in range(0,len(allSimRewards[0]))]; 
		twoSigmaBounds = [0 for i in range(0,len(allSimRewards[0]))]; 
		for i in range(0,len(sampleVariances)):
			suma = 0; 
			for j in range(0,len(allSimRewards)):
				suma += (allSimRewards[j][i] - averageRewards[i])**2; 
			sampleVariances[i] = suma/len(allSimRewards); 
			twoSigmaBounds[i] = sqrt(sampleVariances[i])*2; 

		print('Sample Mean:'+str(averageRewards[len(allSimRewards[0])-1]));
		print('Sample Variance:' + str(sampleVariances[len(allSimRewards[0])-1])); 
		#plot figure
		time = [i for i in range(0,len(allSimRewards[0]))]; 
		plt.figure();
		plt.errorbar(time,averageRewards,yerr=twoSigmaBounds); 
		plt.xlabel('Simulation Step'); 
		plt.title('Average Simulation Reward with Error Bounds for ' + str(len(allSimRewards)) + ' simulations.'); 
		plt.ylabel('Reward'); 
		plt.ylim([-150,200])
		plt.show(); 

	def ani(self,bels,allX0,allX1,numFrames = 20):
		fig, ax = plt.subplots()
		a = np.linspace(0,0,num = 100); 
		xlabel = 'Cop Position';
		ylabel = 'Belief of Robber Position';
		title = 'Belief Animation';

		images = [];

		for t in range(0,numFrames):
		 	if t != 0:
				ax.cla(); 
			 
				
			[x,y,c] = bels[t].plot2D(vis = False); 
			'''
			ax.contourf(x,y,c,cmap = 'viridis'); 
			
			single = ax.scatter(allX0[t],allX1[t],color = 'white', s = 50);
			cop = ax.scatter(allX0[t],-0.25,color = 'blue',s = 50);  
			robber = ax.scatter(-0.25,allX1[t],color = 'red',s = 50); 
			ax.set_xlabel(xlabel); 
			ax.set_ylabel(ylabel);
			ax.set_title(title);
	    	fig.savefig('../tmp/img' + str(t) + ".png"); 
	    	#print('../tmp/img' + str(t) + ".png")
	    	'''
			suma = [0]*len(c); 
			for i in range(0,len(c)):
				for j in range(0,len(c[i])):
					suma[i] += c[j][i]; 

			sumaA = sum(suma); 
			for i in range(0,len(suma)):
				suma[i] = suma[i]/sumaA; 

			ax.plot(x,suma); 
			col = 'blue'; 
			if(abs(allX0[t] - allX1[t]) <= 0.5):
				col = 'green'; 
			cop = ax.scatter(allX0[t],0,color = col,s = 400);
			robber = ax.scatter(allX1[t],max(suma)/2,color = 'red',s = 400);
			ax.set_ylim([0,max(suma)]); 
			ax.set_xlim([0,5]);
			ax.set_xlabel(xlabel); 
			ax.set_ylabel(ylabel);
			ax.set_title(title);
			fig.savefig('../tmp/img' + str(t) + ".png"); 
			plt.pause(0.5)
			
		for k in range(0,numFrames-1):
			fname = "../tmp/img%d.png" %k
			#print(fname); 
			img = mgimg.imread(fname); 
			imgplot = plt.imshow(img); 
			images.append([imgplot]); 


		#fig = plt.figure(); 
		my_ani = animation.ArtistAnimation(fig,images,interval = 20); 
		my_ani.save("../Results/animation.gif",fps = 2)
		#plt.show(); 

	def signal_handler(self,signal, frame):
		print("Stopping Policiy Generation and printing to file"); 
		self.exitFlag = True; 

	



if __name__ == "__main__":

	#Files
	belSave = '../beliefs/interceptBeliefs4.npy'; 
	belLoad = '../../beliefs/interceptBeliefs3.npy';
	alsave = '../policies/interceptAlphas7.npy'; 
	alLoad = '../../policies/interceptAlphas4.npy'; 

	'''	
	********
	Alphas:
	1: Works for initialPose=[2,4], with zdist at 1; 
	2: Works for zdist 0.5 and r at 0.5, for [2,4], might be multi-pose
	3: Run with Belief 3, works alright, could be better...
	4: Works for all starting poses
	5: Generated for moving robber, for all starting poses
	6: BAD. Decreasing Discounts is BADDD. Discount = 0.9, with discountDep = 0.01
	7: 

	Beliefs:
	1: Decent span, works with alpha 1 at initial [2,4]
	2: Works with alpha 2 at initial [2,4] really well
	3: Created for all start positions

	********
	'''


	#Flips and switches
	sol = False; 
	iterations = 500; 
	decreaseDiscount = False;
	decreasePer = 0.01; 

	sim = True; 
	simRand = False; 
	numStep = 100; 
	humanInput = False;
	mdpPolicy = False;  
	greedySim = False; 
	mulSim = True; 
	simCount = 100; 
	
	#usually around 10
	belMaxMix = 5;

	plo = False; 


	a = InterceptTestGenerator(beliefFile = belLoad,dep = decreaseDiscount,depFac = decreasePer); 
	signal.signal(signal.SIGINT, a.signal_handler);

	if(sol):
		a.solve(N = iterations,alsave = alsave,verbose = True); 
	if(sim):
		if(not simRand and not mulSim):
			inPose = [random.randint(0,5),random.randint(0,5)]; 
			[allB,allX0,allX1,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,numSteps = numStep,belSave = belSave,MDP = mdpPolicy,human = humanInput,greedy=greedySim,randSim = simRand,beliefMaxMix = belMaxMix); 
			
			#plt.plot(allReward); 

			#plt.show(); 
			dist = []; 
			for i in range(0,len(allX0)):
				dist.append(allX1[i]-allX0[i]); 
			plt.plot(allX0,'b-',label = 'Cop'); 
			plt.plot(allX1,'r-',label = 'Robber');
			plt.legend( loc='upper right' ) 
			 
			plt.show(); 

			plt.plot(dist); 
			axes = plt.gca(); 
			axes.set_ylim([-5,5]); 
			plt.axhline(y=.5, xmin=0, xmax=100, linewidth=2, color = 'k')
			plt.axhline(y=-.5, xmin=0, xmax=100, linewidth=2, color = 'k')
			plt.xlabel('Simulation Step'); 
			plt.ylabel('Difference: Robber-Cop');
			plt.title('Robot Position Difference with detection zones'); 
			plt.show();


			a.ani(allB,allX0,allX1,numFrames = numStep); 

			'''
			for i in range(0,len(allX0)):
				print(allX0[i],allX1[i]); 
				print(allAct[i]);
				if(abs(allX0[i]-allX1[i]) <= 1):
					print("Detection"); 
			'''
				
		elif(not simRand):
			#run simulations
			allSimRewards = []; 
			if(mdpPolicy):
				a.MDPValueIteration(); 
			for i in range(0,simCount):
				inPose = [random.randint(1,4),random.randint(1,4)];
				print("Starting simulation: " + str(i+1) + " of " + str(simCount) + " with initial position: " + str(inPose));
				[allB,allX0,allX1,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,numSteps = numStep,greedy = greedySim,human = humanInput, mul = mulSim,MDP = mdpPolicy,belSave = belSave,randSim = simRand,beliefMaxMix = belMaxMix,verbose = False); 
				allSimRewards.append(allReward); 
				print("Simulation complete. Reward: " + str(allReward[numStep-1])); 
			a.plotRewardErrorBounds(allSimRewards); 


		else:
			a.simulate(policy = alLoad,initialPose = [2,4],numSteps = numStep,belSave = belSave,randSim = simRand,beliefMaxMix = belMaxMix); 



	if(plo):
		pol = np.load(alLoad); 

		for i in range(0,len(pol)):
			print(pol[i].action); 
			pol[i].plot2D();
	
