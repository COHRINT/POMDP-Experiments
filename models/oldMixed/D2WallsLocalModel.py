from __future__ import division
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


'''
****************************************************
File: D2WallsLocalModel.py
Written By: Luke Burks
January 2017

Container Class for problem specific models
Model: 2D Robot localization problem with
a wall

Bounds from 0 to 5 on both dimensions, 
with a vertical wall from 0-3 at 2.5. 
Reward coded wall


'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



class D2WallsLocalModel:

	def __init__(self,fileNamePrefix = 'tmpModel'):
		self.fileNamePrefix = fileNamePrefix; 
		self.walls = []; 

	#Problem Specific
	def MDPValueIteration(self,gen = True,numIter = 100,maxMix = 20,visualize = True):
		if(gen):
			self.buildReward(gen = False); 
			#Intialize Value function
			self.ValueFunc = copy.deepcopy(self.r[0]); 
			#for g in self.ValueFunc.Gs:
				#g.weight = -1000000; 

			fig,ax = plt.subplots(1); 

			comparision = GM(); 
			var = (np.identity(2)).tolist(); 
			comparision.addG(Gaussian([1,0],var,1)); 

			uniform = GM(); 
			for i in range(0,5):
				for j in range(0,5):
					uniform.addG(Gaussian([i,j],(np.identity(2)*1000).tolist(),1)); 

			count = 0; 

			#until convergence
			flag = False;
			while(not self.ValueFunc.fullComp(comparision) and count < numIter):
				print("Value Function Step: " + str(count) + " of " + str(numIter)); 
				comparision = copy.deepcopy(self.ValueFunc); 
				count += 1;
				#print(count); 
				maxVal = -10000000; 
				maxGM = GM(); 
				for a in range(0,len(self.delA)):
					suma = GM(); 
					for g in self.ValueFunc.Gs:
						mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist(); 
						var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
						suma.addG(Gaussian(mean,var,g.weight));  
					suma.addGM(self.r[a]); 
					tmpVal = self.continuousDot(uniform,suma); 
					if(tmpVal >= maxVal):
						maxVal = tmpVal; 
						maxGM = copy.deepcopy(suma); 

				maxGM.scalerMultiply(self.discount); 
				maxGM = maxGM.kmeansCondensationN(maxMix);
				#maxGM.condense(maxMix);  
				self.ValueFunc = copy.deepcopy(maxGM); 

				 
				if(visualize):
					[x,y,c] = self.ValueFunc.plot2D(vis = False);  
					
					minim = np.amin(c); 
					maxim = np.amax(c); 

					#print(minim,maxim); 
					levels = np.linspace(minim,maxim); 
					ax.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
					

					plt.pause(0.01); 

			print("MDP Value Iteration Complete");
			f = open("../policies/" + self.fileNamePrefix + "MDP.npy","w"); 
			np.save(f,self.ValueFunc);
			#file = "../policies/MDPPolicy"; 
			#self.ValueFunc.printGMArrayToFile([self.ValueFunc],file); 
		else:
			self.ValueFunc = np.load("../policies/" + self.fileNamePrefix + "MDP.npy").tolist(); 
			#file = "../policies/MDP4DIntercept"; 
			#tmp = GM(); 
			#self.ValueFunc = tmp.readGMArray4D(file)[0]; 


	def solveQ(self,gen = True):
		if(gen):
			self.Q =[0]*len(self.delA); 
			V = self.ValueFunc; 
			for a in range(0,len(self.delA)):
				self.Q[a] = GM(); 
				for i in range(0,V.size):
					mean = (np.matrix(V.Gs[i].mean)-np.matrix(self.delA[a])).tolist(); 
					var = (np.matrix(V.Gs[i].var) + np.matrix(self.delAVar)).tolist()
					self.Q[a].addG(Gaussian(mean,var,V.Gs[i].weight)); 
				self.Q[a].addGM(self.r[a]); 
			f = open("../policies/" + self.fileNamePrefix + "QMDP.npy","w"); 
			np.save(f,self.Q);
		else:
			self.Q = np.load("../policies/" + self.fileNamePrefix + "QMDP.npy"); 

	#Problem specific
	def buildTransition(self):
		self.bounds = [[0,5],[0,5]]; 
		self.delAVar = (np.identity(2)*0.01).tolist(); 
		#self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]; 
		self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0]]; 
		self.discount = 0.95; 

		#set wall line segments
		self.walls = [[[2.5,-1],[2.5,2]],[[0,0],[0,5]],[[0,5],[5,5]],[[5,0],[5,5]],[[0,0],[5,0]]]; 
		#self.walls = []; 

	#Problem Specific
	def buildObs(self,gen=True):
		#Quadrant model? Or maybe just halfs? 
		#Hopefully this won't matter if I can just run the MDP and QMDP
		#Maybe just useless obs???

		if(gen):
			self.pz = [0]*1; 
			for i in range(0,1):
				self.pz[i] = GM(); 
			var = (np.identity(2)*1).tolist(); 
			#Left Half
			for i in range(-1,7):
				for j in range(-1,7):
					self.pz[0].addG(Gaussian([i,j],var,1));
							
			print('Plotting Observation Models'); 
			for i in range(0,len(self.pz)):
				#self.pz[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Obs:' + str(i)); 
				[x,y,c] = self.pz[i].plot2D(vis=False); 
				plt.contourf(x,y,c,cmap='viridis'); 
				plt.colorbar(); 
				plt.show(); 

			print('Condensing Observation Models'); 
			for i in range(0,len(self.pz)):
				#self.pz[i] = self.pz[i].kmeansCondensationN(64);
				self.pz[i].condense(15);  

			print('Plotting Condensed Observation Models'); 
			for i in range(0,len(self.pz)):
				#self.pz[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Obs:' + str(i)); 
				[x,y,c] = self.pz[i].plot2D(vis=False); 
				plt.contourf(x,y,c,cmap='viridis'); 
				plt.colorbar(); 
				plt.show(); 


			f = open("../models/"+ self.fileNamePrefix + "OBS.npy","w"); 
			np.save(f,self.pz);
			#file = '../models/obsAltModel4DIntercept'; 
			#self.pz[0].printGMArrayToFile(self.pz,file); 
		else:
			self.pz = np.load("../models/"+ self.fileNamePrefix + "OBS.npy").tolist(); 
			#file = '../models/obsModel4DIntercept'; 
			#tmp = GM(); 
			#self.pz = tmp.readGMArray4D(file); 
			
	#Problem Specific
	def buildReward(self,gen = True):
		if(gen):

			#walls = [[2.5,0],[2.5,.5],[2.5,1],[2.5,1.5],[2.5,2],[2.5,2.5],[2.5,3]]; 
			walls = []; 
			
			for i in range(-10,110):
				if(i<=21):
					walls.append([2.5,i/10]); 
					#sdljf = 0; 

			
			self.r = [0]*len(self.delA);
			self.rNegs = [0]*len(self.delA); 

			for i in range(0,len(self.r)):
				self.r[i] = GM();
			for i in range(0,len(self.rNegs)):
				self.rNegs[i] = GM();  

			var = (np.identity(2)*0.05).tolist(); 
			var2 = (np.identity(2)*0.25).tolist(); 

			#Add Reward in far corner
			for i in range(0,len(self.r)):
				m = (np.array([4.5,0.5]) - np.array(self.delA[i])).tolist();
				self.r[i].addG(Gaussian(m,var2,200));

			#Add Wall Penalty
			for w in walls:
				for i in range(0,len(self.r)):
					m = (np.array(w) - np.array(self.delA[i])).tolist();
					self.rNegs[i].addG(Gaussian(m,var,-100)); 
			

			#print('Plotting Reward Model'); 
			#for i in range(0,len(self.r)):
				#self.r[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 

			print('Condensing Reward Model');
			for i in range(0,len(self.r)):
				self.r[i] = self.r[i].kmeansCondensationN(k = 5);
				self.rNegs[i] = self.rNegs[i].kmeansCondensationN(k=5); 
				self.r[i].addGM(self.rNegs[i]); 


			print('Plotting Condensed Reward Model'); 
			for i in range(0,len(self.r)):
				#self.r[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 
				[x,y,c] = self.r[i].plot2D(vis = False);  
	
				minim = np.amin(c); 
				maxim = np.amax(c); 

				#print(minim,maxim); 
				levels = np.linspace(minim,maxim); 
				plt.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
				plt.title('Reward for action: ' + str(i));
				plt.xlabel('Robot X'); 
				plt.ylabel('Robot Y'); 
				plt.show(); 


			f = open("../models/"+ self.fileNamePrefix + "REWARD.npy","w"); 
			np.save(f,self.r);
			#file = '../models/rewardModel4DIntercept'; 
			#self.r.printGMArrayToFile([self.r],file);
		else:
			self.r = np.load("../models/"+ self.fileNamePrefix + "REWARD.npy").tolist();
			#file = '../models/rewardModel4DIntercept'; 
			#tmp = GM(); 
			#self.r = tmp.readGMArray4D(file)[0]; 



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


if __name__ == '__main__':
	fileNamePrefix = 'D2WallsLocalModel'; 
	a = D2WallsLocalModel(fileNamePrefix); 
	a.buildTransition(); 
	a.buildObs(gen = False); 
	a.buildReward(gen = False); 
	a.MDPValueIteration(gen = False,numIter = 40,maxMix = 100); 
	#a.newMDPValueIteration(gen = True,numIter = 100,maxMix = 80); 
	a.solveQ(gen = False);

	'''
	[x,y,c] = a.r[0].plot2D(vis=False); 
	lev = np.linspace(np.amin(c),np.amax(c),num=1000);
	plt.contourf(x,y,c,levels =lev,cmap='viridis'); 
	plt.colorbar(); 
	plt.show(); 
	'''

	[x,y,c] = a.ValueFunc.plot2D(vis = False);  
	
	minim = np.amin(c); 
	maxim = np.amax(c); 

	#print(minim,maxim); 
	levels = np.linspace(minim,maxim); 
	plt.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
	plt.title('Value Function in Continous Space');
	plt.xlabel('Robot X'); 
	plt.ylabel('Robot Y'); 
	plt.show(); 



