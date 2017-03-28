
from __future__ import division
'''
****************************************************
File: PolicyTranslator.py
Written By: Luke Burks
December 2016

This is intended as a template for POMDP policy 
translators. Ideally all problem specific bits
will have been removed

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


from D4NoWallsModels import D4NoWallsModel
from D4WallsModels import D4WallsModel
from D2WallsLocalModel import D2WallsLocalModel
from D2DiffsModel import D2DiffsModel
from D2DiffsModelSoftmax import D2DiffsModelSoftmax


class PolicyTranslatorSoftmax:

	def __init__(self, fileNamePrefix, gen=False,mdpLoad = False,qLoad = False,humObs = True,alphaLoad = None):
		#Initialize exit flag
		self.exitFlag = False; 
		self.b = None; 

		#Grab Modeling Code
		allMod = D2DiffsModelSoftmax(fileNamePrefix); 

		#Build Transition Model
		allMod.buildTransition(); 
		self.delA = allMod.delA; 
		self.delAVar = allMod.delAVar; 
		self.bounds = allMod.bounds; 

		#Get walls, if there are any
		self.walls = allMod.walls; 

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
		self.bounds = allMod.bounds; 
		

		#QMDP Generation
		if(qLoad == True):
			allMod.solveQ(gen = gen); 
			self.Q = allMod.Q; 			

		if(mdpLoad):
			allMod.MDPValueIteration(gen = gen); 
			self.ValueFunc = allMod.ValueFunc; 

		#Initialize Gamma
		self.loadPolicy(alphaLoad); 



	def loadPolicy(self,fileName):
		self.Gamma = np.load(fileName); 

	def getAction(self,b):
		act = self.Gamma[np.argmax([self.continuousDot(j,b) for j in self.Gamma])].action;
		return act; 

	def getSecondaryAction(self,b,exclude):
		sG = [];
		for g in self.Gamma:
			if(g.action not in exclude):
				sG.append(g); 
		act = sG[np.argmax([self.continuousDot(j,b) for j in sG])].action;
		return act; 

	def getGreedyAction(self,b,x):
		#cut = b.slice2DFrom4D(retGS=True,vis=False); 
		cut = b;
		MAP = cut.findMAP2D();
		#cop = [x[0],x[1]]; 
		#rob = [MAP[0],MAP[1]]; 
		xdist = x[0]; 
		ydist = x[1]; 

		if(abs(xdist)>abs(ydist)):
			if(xdist > 0):
				act = 0; 
			else:
				act = 1; 
		else:
			if(ydist > 0):
				act = 2; 
			else:
				act = 3; 

		return act; 

	def getMDPAction(self,x):
		maxVal = -100000000000; 
		maxGM = GM();
		bestAct = 0;  
		for a in range(0,len(self.delA)):
			suma = GM(); 
			for g in self.ValueFunc.Gs:
				mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist(); 
				var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
				suma.addG(Gaussian(mean,var,g.weight));  
			suma.addGM(self.r[a]); 
			
			tmpVal = suma.pointEval(x); 
			if(tmpVal > maxVal):
				maxVal = tmpVal; 
				maxGM = suma;
				bestAct = a; 
		return bestAct; 

	def getMDPSecondaryAction(self,x,exclude):
		maxVal = -100000000000; 
		maxGM = GM();
		bestAct = 0;  
		for a in range(0,len(self.delA)):
			if(a in exclude):
				continue; 
			suma = GM(); 
			for g in self.ValueFunc.Gs:
				mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist(); 
				var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
				suma.addG(Gaussian(mean,var,g.weight));  
			suma.addGM(self.r[a]); 
			
			tmpVal = suma.pointEval(x); 
			if(tmpVal > maxVal):
				maxVal = tmpVal; 
				maxGM = suma;
				bestAct = a; 
		return bestAct; 

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


	def beliefUpdate(self,b,a,o,maxMix = 10):

		btmp = GM(); 
		btmp1 = GM(); 
		for j in b.Gs:
			mean = (np.matrix(j.mean) + np.matrix(self.delA[a])).tolist()[0]; 
			var = (np.matrix(j.var) + np.matrix(self.delAVar)).tolist(); 
			weight = j.weight; 
			btmp1.addG(Gaussian(mean,var,weight)); 
		btmp = self.pz.runVBND(btmp1,o); 
		
		#btmp.condense(maxMix);
		btmp = btmp.kmeansCondensationN(maxMix);  
		btmp.normalizeWeights();

		return btmp; 

	def distance(self,x1,y1,x2,y2):
		dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2); 
		dist = math.sqrt(dist); 
		return dist; 

	def generateBeliefs(self,numPose = 10, numSteps = 10,belSave = 'tmpbelSave.npy',beliefMaxMix = 10):

		allB = [];  
		allX = [];  
		allXInd = [0]*len(self.delA[0]); 
		for i in range(0,len(self.delA[0])):
			allXInd[i] = []; 

		#TODO: Bad, remove, not general
		allPose = []; 
		for i in range(-10,10):
			for j in range(-10,10):
				allPose.append([i,j]);

		'''
		for i in range(0,numPose):
			
			b = GM(); 
			mean = [0]*len(self.delA[0]); 
			var = [[0 for k in range(0,len(self.delA[0]))] for j in range(0,len(self.delA[0]))]; 
			for k in range(0,len(self.delA[0])):
				mean[k] = random.random()*self.bounds[k][1] + self.bounds[k][0]; 
				var[k][k] = random.random()*3;  
			x = mean; 
			b.addG(Gaussian(mean,var,1)); 
			for h in range(0,numSteps):
				act = random.randint(0,len(self.delA)-1); 

				x = np.random.multivariate_normal(np.array(x)+np.array(self.delA[act]),self.delAVar,size =1)[0].tolist();
				
				for i in range(0,len(x)):
					x[i] = max(self.bounds[i][0],x[i]); 
					x[i] = min(self.bounds[i][1],x[i]); 

				ztrial = [0]*len(self.pz); 

				for i in range(0,len(self.pz)):
					ztrial[i] = self.pz[i].pointEval(x); 
				z = ztrial.index(max(ztrial)); 


				b = self.beliefUpdate(b,act,z,beliefMaxMix); 
				

				allB.append(b);
				allX.append(x);
				for i in range(0,len(x)):
					allXInd[i].append(x[i]);  
		'''
		#TODO: Remove, bad, not general

		for i in range(0,len(allPose)):
			
			b = GM(); 
			mean = copy.deepcopy(allPose[i]);  
			x = allPose[i];
			var = (np.identity(2)*0.01).tolist();  
			b.addG(Gaussian(mean,var,1)); 
			for h in range(0,numSteps):
				act = random.randint(0,len(self.delA)-1); 

				x = np.random.multivariate_normal(np.array(x)+np.array(self.delA[act]),self.delAVar,size =1)[0].tolist();
				
				for i in range(0,len(x)):
					x[i] = max(self.bounds[i][0],x[i]); 
					x[i] = min(self.bounds[i][1],x[i]); 

				ztrial = [0]*self.pz.size; 

				for i in range(0,self.pz.size):
					ztrial[i] = self.pz.pointEval2D(i,x);  
				z = ztrial.index(max(ztrial)); 


				b = self.beliefUpdate(b,act,z,beliefMaxMix); 
				

				allB.append(b);
				allX.append(x);
				for i in range(0,len(x)):
					allXInd[i].append(x[i]); 


		f = open(belSave,"w"); 
		np.save(f,allB); 
		print("Total Number of Beliefs: " + str(len(allB))); 
		for i in range(0,len(x)):
			print("State " + str(i) + ", Max:" + str(max(allXInd[i])) + ", Min: " + str(min(allXInd[i])) + ", Average:" +  str(sum(allXInd[i])/float(len(allXInd[i])))); 


	def simulate(self,policy = "interceptAlphasTemp.npy",initialPose = [1,4],initialBelief = None, numSteps = 20,mul = False,QMDP = False,MDP = False,mdpGen = True,human = False,greedy = False,randSim = False,altObs = False,belSave = 'tmpbelSave.npy',beliefMaxMix = 10,verbose = True):

		if(initialBelief == None):
			b = GM(); 
			mean = [0]*len(self.delA[0]); 
			var = [[0 for k in range(0,len(self.delA[0]))] for j in range(0,len(self.delA[0]))]; 
			for k in range(0,len(self.delA[0])):
				mean[k] = random.random()*(self.bounds[k][1]-self.bounds[k][0]) + self.bounds[k][0]; 
				var[k][k] = random.random()*10;  
			b.addG(Gaussian(mean,var,1));
		else:
			b = initialBelief; 
			

		x = initialPose; 
		allX = []; 
		allX.append(x); 
		allXInd = [0]*len(self.delA[0]); 
		for i in range(0,len(self.delA[0])):
			allXInd[i] = [x[i]]; 

		reward = 0; 
		allReward = [0]; 
		allB = []; 
		allB.append(b); 

		allAct = []; 



		for count in range(0,numSteps):

			if(verbose):
				print('Step: ' + str(count)); 

			if(human):
				#stuff...
				dslfkj = 0; 
			elif(greedy):
				act = self.getGreedyAction(b,x); 
			elif(MDP):
				act = self.getMDPAction(x); 
			elif(QMDP):
				act = self.getQMDPAction(b);
			else:
				act = self.getAction(b);

			# if(act == 0):
			# 	act = 1; 
			# elif(act == 1):
			# 	act = 0; 

			#TODO: Random Flag, remove
			#act = random.randint(0,len(self.delA)-1)

			xsave = copy.deepcopy(x); 
			excluedActions = []; 

			x = np.random.multivariate_normal(np.array(x)+np.array(self.delA[act]),self.delAVar,size =1)[0].tolist();
			
			#bound the movement
			for i in range(0,len(x)):
				x[i] = max(self.bounds[i][0],x[i]); 
				x[i] = min(self.bounds[i][1],x[i]);

			 

			ztrial = [0]*self.pz.size; 

			for i in range(0,self.pz.size):
				ztrial[i] = self.pz.pointEval2D(i,x);  
			z = ztrial.index(max(ztrial)); 

			'''
			if(not z==4):
				tmpb = GM(); 
				for i in range(0,4):
					tmpb.addGM(self.beliefUpdate(b,act,i,beliefMaxMix)); 
				b = tmpb; 
				b = b.kmeansCondensationN(k=beliefMaxMix);  
			else:
				b = self.beliefUpdate(b,act,z,beliefMaxMix);
			'''
			b = self.beliefUpdate(b,act,z,beliefMaxMix);

			#print(b.size); 

			allB.append(b);
			allX.append(x);
			allAct.append(act); 
			for i in range(0,len(x)):
				allXInd[i].append(x[i]);  

			reward += self.r[act].pointEval(x); 
			allReward.append(reward); 
			


		
		allAct.append(-1);
		if(verbose):
			print("Simulation Complete. Accumulated Reward: " + str(reward));  
		return [allB,allX,allXInd,allAct,allReward]; 

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
		plt.show(); 

	def ani(self,bels,allX0,allX1,allX2,allX3,numFrames = 20):
		fig, ax = plt.subplots()
		a = np.linspace(0,0,num = 100); 
		xlabel = 'Diff X Position';
		ylabel = 'Diff Y Position';
		title = 'Belief Animation';

		images = [];

		'''
		wall1 = [[2,0],[2,.5],[2,1],[2,1.5],[2,2],[1.5,2],[1,2]]; 
		wall2 = [[3,0],[3,.5],[3,1],[3,1.5],[3,2],[3.5,2],[4,2]]; 
		wall3 = [[2,5],[2,4.5],[2,4],[2,3.5],[2,3],[1.5,3],[1,3]]; 
		wall4 = [[3,5],[3,4.5],[3,4],[3,3.5],[3,3],[3.5,3],[4,3]]; 

		walls = wall1+wall2+wall3+wall4;

		wallsX = []; 
		wallsY = []; 
		for w in walls:
			wallsX.append(w[0]); 
			wallsY.append(w[1]); 
		'''

		for t in range(0,numFrames):
		 	if t != 0:
				ax.cla(); 
			 
				
				#[x,y,c] = bels[t].slice2DFrom4D(vis = False); 
				[x,y,c] = bels[t].plot2D(low = [-10,-10],high = [10,10],vis = False); 
				ax.contourf(x,y,c,cmap = 'viridis'); 
				
				col = 'b'; 
				#if(self.distance(allX0[t],allX1[t],allX2[t],allX3[t]) <= 1):
				if(abs(allX0[t]) <=1 and abs(allX1[t]) <=1):
					col = 'g'
				dist = ax.scatter(allX0[t],allX1[t],color = col,s = 100);  
				#robber = ax.scatter(allX2[t],allX3[t],color = 'red',s = 100); 
				#walls = ax.scatter(wallsX,wallsY,color = 'black',s=300); 
				ax.set_xlabel(xlabel); 
				ax.set_ylabel(ylabel);
				ax.set_title(title);
				fig.savefig('../tmp/img' + str(t) + ".png"); 
				#print('../tmp/img' + str(t) + ".png")
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

	def plotAllSlices(self,a,title):
		fig,ax = plt.subplots(2,2); 
		[x1,y1,c1] = a.slice2DFrom4D(vis=False,dims=[0,2]); 
		ax[0,0].contourf(x1,y1,c1,cmap = 'viridis'); 
		ax[0,0].set_title('Cop X with Robber X'); 

		[x2,y2,c2] = a.slice2DFrom4D(vis=False,dims=[0,3]); 
		ax[0,1].contourf(x2,y2,c2,cmap = 'viridis'); 
		ax[0,1].set_title('Cop X with Robber Y');

		[x3,y3,c3] = a.slice2DFrom4D(vis=False,dims=[1,2]); 
		ax[1,0].contourf(x3,y3,c3,cmap = 'viridis'); 
		ax[1,0].set_title('Cop Y with Robber X'); 

		[x4,y4,c4] = a.slice2DFrom4D(vis=False,dims=[1,3]); 
		ax[1,1].contourf(x4,y4,c4,cmap = 'viridis'); 
		ax[1,1].set_title('Cop Y with Robber Y'); 

		fig.suptitle(title); 
		plt.show(); 

	def lineSegOrient(self,a,b,c):
		#asks if three points are clockwise oriented
		tau = (b[1]-a[1])*(c[0]-b[0]) - (c[1]-b[1])*(b[0]-a[0]); 

		if(tau < 0):
			return False;
		else:
			return True;

	def checkIntercept(self,a,b):
		o1 = self.lineSegOrient(a[0],a[1],b[0]); 
		o2 = self.lineSegOrient(a[0],a[1],b[1]); 
		o3 = self.lineSegOrient(b[0],b[1],a[0]); 
		o4 = self.lineSegOrient(b[0],b[1],a[1]); 

		if(o1 != o2 and o3 != o4):
			return True; 
		else:
			return False; 

	def crossedWall(self,x1,x2):

		for w in self.walls:
			if(self.checkIntercept([x1,x2],w)):
				return True;
		return False; 



if __name__ == "__main__":
	



	#Files
	belSave = '../beliefs/DiffsPolicyBeliefs2.npy'; 
	alLoad = '../policies/DiffsSoftmaxAlphas2.npy'; 

	'''	
	********
	Beliefs:
	1: Fine
	********
	'''




	#Flips and switches
	dataNumber = 4; 

	#controls obs and reward generation
	generate = False; 

	#Simulation Controls
	sim = True; 
	numStep = 100;

	simRand = False;
	randPose = 100;  
	randStep = 1; 
	 

	humanInput = False;

	mdpPolicy = False;  
	mdpGen = False; 

	greedySim = False; 

	qmdp = False; 

	mulSim = True; 
	simCount = 100; 
	
	#ususally around 10
	belMaxMix = 20;

	hObs = False;


	#Plotting Knobs
	plo = False; 


	fileNamePrefix = 'D2DiffsModelSoftmax';

	a = PolicyTranslatorSoftmax(fileNamePrefix = fileNamePrefix,gen = generate,mdpLoad = mdpPolicy,qLoad = qmdp,humObs = hObs,alphaLoad = alLoad); 
	signal.signal(signal.SIGINT, a.signal_handler);
	

	if(sim):
		if(simRand):
			a.generateBeliefs(numPose = randPose,numSteps = randStep,belSave = belSave,beliefMaxMix = belMaxMix); 
		elif(mulSim):
			#run simulations
			allSimRewards = []; 
			for i in range(0,simCount):
				inPose = [0 for j in range(0,len(a.delA[0]))];
				for j in range(0,len(a.delA[0])):
					inPose[j] = random.random()*(a.bounds[j][1]-a.bounds[j][0]) + a.bounds[j][0]; 

				inBel = GM();
				inBel.addG(Gaussian(inPose,[[10,0],[0,10]],1)); 
				print("Starting simulation: " + str(i+1) + " of " + str(simCount) + " with initial position: " + str(inPose));
				[allB,allX,allXInd,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,initialBelief=inBel,numSteps = numStep,belSave = belSave,QMDP = qmdp,MDP = mdpPolicy,human = humanInput,greedy=greedySim,beliefMaxMix = belMaxMix,verbose = False); 
				
				allSimRewards.append(allReward); 

				print("Simulation complete. Reward: " + str(allReward[numStep-1])); 
			

			#save all data
			dataSave = {"Beliefs":allB,"States":allX,"States(Ind)":allXInd,"Actions":allAct,'Rewards':allSimRewards};
			fileSave = '../Results/diffs/' + fileNamePrefix + '_Data' + str(dataNumber) + '.npy';
			f = open(fileSave,'w'); 
			np.save(f,dataSave); 

			a.plotRewardErrorBounds(allSimRewards); 

		else:
			inPose = [0 for i in range(0,len(a.delA[0]))];
			for j in range(0,len(a.delA[0])):
				inPose[j] = random.random()*(a.bounds[j][1]-a.bounds[j][0]) + a.bounds[j][0];  
				
			#inPose = [4.518704850352574, -6.606250448814055]; 
			inBel = GM(); 
			inBel.addG(Gaussian(inPose,[[10,0],[0,10]],1)); 
			[allB,allX,allXInd,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,initialBelief = inBel,numSteps = numStep,belSave = belSave,QMDP = qmdp,MDP = mdpPolicy,human = humanInput,greedy=greedySim,beliefMaxMix = belMaxMix,verbose = True); 
			

			
			fig,ax = plt.subplots(len(a.delA[0]),sharex = True); 
			x = [i for i in range(0,len(allX))]; 
			
			for i in range(0,len(a.delA[0])):
				ax[i].plot(x,allXInd[i],label = ('State:' + str(i)))
				ax[i].set_ylim(a.bounds[i]); 
			 
			#plt.show(); 

			fig2,ax2 = plt.subplots(1,sharex = True);  
			ax2.plot(allXInd[0],allXInd[1]); 
			ax2.set_ylim([-10,10]); 
			ax2.set_xlim([-10,10]); 
			plt.show(); 

			print(allAct)

			fig3,ax3 = plt.subplots(1,sharex=True); 
			for b in allB:
				[x,y,c] = b.plot2D(low = [-10.5,-10.5],high = [10.5,10.5],vis = False); 
				ax3.contourf(x,y,c,cmap='viridis'); 
				ax3.scatter(allX[allB.index(b)][0],allX[allB.index(b)][1],color = 'r'); 
				plt.pause(0.1); 


			#a.ani(allB,allXInd[0],allXInd[1],allXInd[2],allXInd[3],numFrames = numStep); 
			


		
	if(plo):
		for gm in a.Gamma:
			#gm.plot2D(xlabel = 'Robot X',ylabel = 'Robot Y', title = 'Alpha with Action:' + str(gm.action)); 
			[x,y,c] = gm.plot2D(low =[-10,-10],high=[10,10],vis = False);
			minim = np.amin(c); 
			maxim = np.amax(c); 

			#print(minim,maxim); 
			levels = np.linspace(minim,maxim); 
			plt.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
			plt.title('Alpha with Action:' + str(gm.action))
			plt.show(); 

	
