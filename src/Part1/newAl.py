

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "0.1"
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
#cProfile.run('re.compile("foo|bar")','restats'); 

class NewPOMDPSolution:


	def __init__(self,nB = 2,gamma = .95):
		
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

		print("Inititalization Complete");

		


	def solve(self,N = 1,nBranch = 4):
		
		for counter in range(0,N):
			bestAlphas = [[0 for i in range(0,len(self.B[0]))] for j in range(0,len(self.B))]
			Value = [0 for i in range(0,len(self.B))]; 

			for b in self.B:
				bestAlphas[self.B.index(b)] = self.getMaxingAlpha(b); 
				Value[self.B.index(b)] = self.dotProduct(bestAlphas[self.B.index(b)],b); 
			GammaNew = []; 

			#print(""); 
			#print(bestAlphas); 

			BTilde = copy.deepcopy(self.B); 

			while(len(BTilde) > 0):

				b = random.choice(BTilde); 
				BTilde.remove(b); 
				back = self.backup(b); 
				alpha = back[0];  
				act = back[1]; 

				for al in alpha:
					if(self.dotProduct(al,b) < Value[self.B.index(b)]):
						al = bestAlphas[self.B.index(b)]; 

					#remove from Btilde all b for which this alpha is better than its current
					for bprime in BTilde:
						if(self.dotProduct(al,bprime) >= Value[self.B.index(bprime)]):
							BTilde.remove(bprime); 

					GammaNew = GammaNew + [al]; 
			self.Gamma = copy.deepcopy(GammaNew); 

	def dotProduct(self,a,b):
		value = 0; 
		for i in range(0,len(b)):
			value = value+a[i]*b[i]; 
		return value; 


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

	def getTransitionMatrix(self):
		#Builds the state transition model
		a = 0; 

	def getObservationModel(self):
		#Builds the observation model
		a = 0; 

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


	def tao(self,b,a,o):
		pz = self.pz; 
		bprime = [0 for i in range(0,len(self.px))]; 
		for i in range(0,len(bprime)):
			bprime[i] = pz[i][o]; 
			tmp = 0; 
			for j in range(0,len(bprime)):
				tmp += self.px[j][a][i]*b[j]; 
			bprime[i] = bprime[i]*tmp; 
			bprime[i] = self.normalize(bprime); 
		return bprime; 

	



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


def convertVectorToGrid(b):
	a = [[0 for i in range(0,10)] for j in range(0,10)]; 
	for i in range(0,100):
		a[i/10][i%10] = b[i]; 
	return a; 

if __name__ == "__main__":
	a = NewPOMDPSolution(); 
	a.solve(N = 30);  
	
	'''
	f,axarr = plt.subplots(len(a.Gamma),sharex = True); 
	for i in range(0,len(a.Gamma)):
		axarr[i].imshow(convertVectorToGrid(a.Gamma[i])); 
	
	plt.show(); 
	'''
	

