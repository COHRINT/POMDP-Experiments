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
File: D4NoWallsModels.py
Written By: Luke Burks
December 2016

Container Class for problem specific models
Model: 4D Cop v Robber intercept problem 
with no obstructions

'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"


class D4NoWallsModel:

	#Problem Specific
	def MDPValueIteration(self,gen = True):
		if(gen):
			#Intialize Value function
			self.ValueFunc = copy.deepcopy(self.r); 
			for g in self.ValueFunc.Gs:
				g.weight = -1000; 

			comparision = GM(); 
			comparision.addG(Gaussian([1,0,0,0],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],1)); 

			uniform = GM(); 
			for i in range(0,5):
				for j in range(0,5):
					for k in range(0,5):
						for l in range(0,5):
							uniform.addG(Gaussian([i,j,k,l],[[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,4]],1)); 

			count = 0; 

			#until convergence
			while(not self.ValueFunc.comp(comparision) and count < 100):
				print(count); 
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
				maxGM = maxGM.kmeansCondensationN(20); 
				self.ValueFunc = copy.deepcopy(maxGM); 

			print("MDP Value Iteration Complete");
			f = open("../policies/MDPPolicy.npy","w"); 
			np.save(f,self.ValueFunc);
			#file = "../policies/MDPPolicy"; 
			#self.ValueFunc.printGMArrayToFile([self.ValueFunc],file); 
		else:
			self.ValueFunc = np.load("../policies/MDP4DIntercept.npy").tolist(); 
			#file = "../policies/MDP4DIntercept"; 
			#tmp = GM(); 
			#self.ValueFunc = tmp.readGMArray4D(file)[0]; 

	#Problem specific
	def buildTransition(self):
		self.bounds = [[0,5],[0,5],[0,5],[0,5]]; 
		self.delAVar = [[0.0001,0,0,0],[0,0.0001,0,0],[0,0,0.15,0],[0,0,0,0.15]]; 
		self.delA = [[-1,0,0,0],[1,0,0,0],[0,-1,0,0],[0,1,0,0],[0,0,0,0]]; 

	#Problem Specific
	def buildObs(self,gen=True):
		#A front back left right center model
		#0:center
		#1-4: left,right,down,up

		if(gen):
			self.pz = [0]*5; 
			for i in range(0,5):
				self.pz[i] = GM(); 
			var = [[.7,0,0,0],[0,.7,0,0],[0,0,.7,0],[0,0,0,.7]];
			for i in range(-1,7):
				for j in range(-1,7):
					self.pz[0].addG(Gaussian([i,j,i,j],var,1));

			for i in range(-1,7):
				for j in range(-1,7):
					for k in range(-1,7):
						for l in range(-1,7):
							if(i-k>0):
								self.pz[1].addG(Gaussian([i,j,k,l],var,1));
							if(i-k<0):
								self.pz[2].addG(Gaussian([i,j,k,l],var,1)); 
							if(j-l>0):
								self.pz[3].addG(Gaussian([i,j,k,l],var,1)); 
							if(j-l<0):
								self.pz[4].addG(Gaussian([i,j,k,l],var,1)); 
							
			print('Plotting Observation Models'); 
			for i in range(0,len(self.pz)):
				self.plotAllSlices(self.pz[i],title = 'Uncondensed Observation'); 

			print('Condensing Observation Models'); 
			for i in range(0,len(self.pz)):
				self.pz[i] = self.pz[i].kmeansCondensationN(40,lowInit = [-1,-1,-1,-1], highInit = [7,7,7,7]);

			print('Plotting Condensed Observation Models'); 
			for i in range(0,len(self.pz)):
				self.plotAllSlices(self.pz[i],title = 'Condensed Observation'); 



			f = open("../models/obsAltModel4DIntercept.npy","w"); 
			np.save(f,self.pz);
			#file = '../models/obsAltModel4DIntercept'; 
			#self.pz[0].printGMArrayToFile(self.pz,file); 
		else:
			self.pz = np.load("../models/obsAltModel4DIntercept.npy").tolist(); 
			#file = '../models/obsModel4DIntercept'; 
			#tmp = GM(); 
			#self.pz = tmp.readGMArray4D(file); 
			
	#Problem Specific
	def buildReward(self,gen = True):
		if(gen):
			self.r = GM(); 
			var = [[1,0,.7,0],[0,1,0,.7],[.7,0,1,0],[0,.7,0,1]];  
			for i in range(-2,8):
				for j in range(-2,8):
					self.r.addG(Gaussian([i,j,i,j],var,5.6));

			for i in range(-2,8):
				for j in range(-2,8):
					for k in range(-2,8):
						for l in range(-2,8):
							if(abs(i-j) >=2 or abs(k-l) >= 2):
								self.r.addG(Gaussian([i,j,k,l],var,-1)); 

			print('Plotting Reward Model'); 
			self.plotAllSlices(self.r,title = 'Uncondensed Reward');

			print('Condensing Reward Model'); 
			self.r = self.r.kmeansCondensationN(40,lowInit = [-1,-1,-1,-1], highInit = [7,7,7,7])

			print('Plotting Condensed Reward Model'); 
			self.plotAllSlices(self.r,title = 'Condensed Reward');


			f = open("../models/rewardModel4DIntercept.npy","w"); 
			np.save(f,self.r);
			#file = '../models/rewardModel4DIntercept'; 
			#self.r.printGMArrayToFile([self.r],file);
		else:
			self.r = np.load("../models/rewardModel4DIntercept.npy").tolist();
			#file = '../models/rewardModel4DIntercept'; 
			#tmp = GM(); 
			#self.r = tmp.readGMArray4D(file)[0]; 

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
