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


from scipy.stats import norm
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
from softmaxModels import Softmax

#TODO: Include stop-flag criteria


class InterceptTestGenerator:

	def __init__(self,beliefFile = None):
		
		 

		#Initialize exit flag
		self.exitFlag = False; 

		self.buildTransition(); 
		self.buildObs(); 
		self.buildReward(); 
		self.discount = 0.9; 

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
		
		

	
		 



	def solve(self,N,maxMix = 20, finalMix = 50,comb = False, verbose = False, alsave = "interceptAlphasTemp.npy"):

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
			#self.newPreComputeAls(); 


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
				'''
				else:
					index = 0; 
					for h in self.B:
						if(b.comp(h)):
							index = self.B.index(h);
					bestAlphas[index] = al; 
				'''

				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
					if(self.continuousDot(al,bprime) >= Value[self.findB(bprime)]):
						BTilde.remove(bprime); 

				#make sure the alpha doesn't already exist
				addFlag = True; 
				for i in range(0,len(GammaNew)):
					if(al.comp(GammaNew[i])):
						addFlag = False; 
				'''
				#check if the alpha is dominated over the whole space
				#TODO: Generalize this/see if its needed
				dom = [True]*len(GammaNew); 
				res = 100; 
				alPoints = [0]*res; 
				for i in range(0,res):
					alPoints[i] = al.pointEval((5*i)/res); 
				GamPoints = [[0 for i in range(0,res)] for j in range(0,len(GammaNew))]; 
				for i in range(0,res):
					for j in range(0,len(GammaNew)):
						GamPoints[j][i] = GammaNew[j].pointEval((5*i)/res); 

				for i in range(0,len(GammaNew)):
					for j in range(0,res):
						if(alPoints[j] > GamPoints[i][j]):
							dom[i] = False;
							break; 
				if(True in dom):
					addFlag = False; 
				'''

				if(addFlag):
					GammaNew += [al];


			if(comb):
				#Test the results from just combining all the alphas into one big one for each action
				tmp = [0]*len(self.delA); 
				gamcount = [0]*len(self.delA); 
				for i in range(0,len(tmp)):
					tmp[i] = GM([50,50,50,50],[[1,0,.7,0],[0,1,0,.7],[.7,0,1,0],[0,.7,0,1]], 0.0001);
					tmp[i].action = i; 
				for g in GammaNew:
					tmp[g.action].addGM(g);
					gamcount[g.action] = gamcount[g.action] + 1;  
				GammaNew = tmp; 
				
				for i in range(0,len(GammaNew)):
					if(gamcount[i] != 0):
						GammaNew[i].scalerMultiply(1/gamcount[i]); 
			




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

		als1 = [[[0 for i in range(0,len(self.softmax.weights))] for j in range(0,len(self.delA))] for k in range(0,len(G))]; 

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				for o in range(0,len(self.softmax.weights)):
					als1[j][a][o] = GM(); 

					alObs = self.softmax.runVBND(G[j],softClassNum = o); 

					for k in alObs.Gs:
						mean = (np.matrix(k.mean) - np.matrix(self.delA[a])).tolist(); 
						var = (np.matrix(k.var) + np.matrix(self.delAVar)).tolist(); 
						weight = k.weight; 
						als1[j][a][o].addG(Gaussian(mean,var,weight)); 

		self.preAls = als1; 



	def backup(self,b):
		G = self.Gamma; 
		R = self.r; 
		 

		als1 = self.preAls; 
		

		#one alpha for each belief, so one per backup

		
		bestVal = -10000000000; 
		bestAct= 0; 
		bestGM = []; 



		for a in range(0,len(self.delA)):
			suma = GM(); 
			for o in range(0,len(self.softmax.weights)):
				suma.addGM(als1[np.argmax([self.continuousDot(als1[j][a][o],b) for j in range(0,len(als1))])][a][o]); 
			suma.scalerMultiply(self.discount); 
			
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
		#self.delAVar = [[0.0001,0],[0,0.25]]; 
		#self.delA = [[-0.5,0],[0.5,0],[0,0]]; 
		self.delAVar = [[0.0001,0],[0,0.25]]; 
		self.delA = [[-0.5,0],[0.5,0],[0,0]]; 


	def buildObs(self):
		weight = [[-1.3926,1.3926],[-0.6963,0.6963],[0,0]];
		bias = [0,.1741,0]; 
		self.softmax = Softmax(weight,bias); 
		
		
		
	def buildObsSingleGauss(self):
		asdf = 0; 


	def buildReward(self):
		self.r = GM(); 
		var = [[1,.9],[.9,1]];

		for i in range(-1,7):
			self.r.addG(Gaussian([i,i],var,5.6250));
		for i in range(1,13):
			for j in range(-1,7):
				self.r.addG(Gaussian([j,j+i],var,-1)); 
				self.r.addG(Gaussian([j,j-i],var,-1));


		self.r.condense(25); 


	def beliefUpdate(self,b,a,o,maxMix = 10):

		btmp = GM(); 
		btmp1 = GM(); 
		for j in b.Gs:
			mean = (np.matrix(j.mean) + np.matrix(self.delA[a])).tolist()[0]; 
			var = (np.matrix(j.var) + np.matrix(self.delAVar)).tolist(); 
			weight = j.weight; 
			btmp1.addG(Gaussian(mean,var,weight)); 
		#btmp = btmp1.runVB(self.soft_weight,self.soft_bias,self.soft_alpha,self.soft_zeta_c,softClassNum = o); 
		btmp = self.softmax.runVBND(btmp1,o); 
		
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
						elif(abs(x[0]-x[1] >= 0.5)):
							z = 2; 
						else:
							z=1;

						b = self.beliefUpdate(b,act,z,beliefMaxMix); 
						if(z == 2):
							b.addGM(self.beliefUpdate(b,act,0,beliefMaxMix));
						elif(z == 0):
							b.addGM(self.beliefUpdate(b,act,2,beliefMaxMix));
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
				elif(abs(x[0]-x[1] >= 0.5)):
					z = 2; 
				else:
					z=1;

				#if(not MDP):
				b = self.beliefUpdate(b,act,z,beliefMaxMix); 
				'''
				if(z == 2):
					b.addGM(self.beliefUpdate(b,act,0,beliefMaxMix));
				elif(z == 0):
					b.addGM(self.beliefUpdate(b,act,2,beliefMaxMix));
				'''
				
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
	belSave = '../beliefs/2DsoftInterceptBeliefs3.npy'; 
	belLoad = '../../beliefs/2DsoftInterceptBeliefs2.npy';
	alsave = '../policies/2DsoftInterceptAlphas7.npy'; 
	alLoad = '../../policies/2DsoftInterceptAlphas5.npy'; 

	'''	
	********
	Alphas:
	5: softmax in 2D, left right and center
	6: MMS detect/no-detect, good
	

	Beliefs:
	1: good for 3 obs
	2: MMS

	********
	'''


	#Flips and switches
	sol = False; 
	iterations = 50; 

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


	a = InterceptTestGenerator(beliefFile = belLoad); 
	signal.signal(signal.SIGINT, a.signal_handler);

	if(sol):
		a.solve(N = iterations,alsave = alsave,comb = False,verbose = True); 
	if(sim):
		if(not simRand and not mulSim):
			#inPose = [random.randint(0,5),random.randint(0,5)]; 
			inPose = [random.randint(0,5),random.randint(0,5)]; 
			[allB,allX0,allX1,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,numSteps = numStep,belSave = belSave,MDP = mdpPolicy,human = humanInput,greedy=greedySim,randSim = simRand,beliefMaxMix = belMaxMix); 
			
			#plt.plot(allReward); 

			#plt.show(); 
			'''
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
			'''

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
				inPose = [random.randint(0,5),random.randint(0,5)]; 
				print("Starting simulation: " + str(i+1) + " of " + str(simCount) + " with initial position: " + str(inPose));
				[allB,allX0,allX1,allAct,allReward]= a.simulate(policy = alLoad,initialPose = inPose,numSteps = numStep,greedy = greedySim,human = humanInput, mul = mulSim,MDP = mdpPolicy,belSave = belSave,randSim = simRand,beliefMaxMix = belMaxMix,verbose = False); 
				allSimRewards.append(allReward); 
				print("Simulation complete. Reward: " + str(allReward[numStep-1])); 
			a.plotRewardErrorBounds(allSimRewards); 


		else:
			a.simulate(policy = alLoad,initialPose = [1,0],numSteps = numStep,belSave = belSave,randSim = simRand,beliefMaxMix = belMaxMix); 



	if(plo):
		pol = np.load(alLoad); 
		for i in range(0,len(pol)):
			print(pol[i].action);  
			pol[i].plot(low = 0,high = 5);
			

	
