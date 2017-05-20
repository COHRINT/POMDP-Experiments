'''
######################################################

File: testMCTS.py
Author: Luke Burks
Date: April 2017

Implements the Monte Carlo Tree Search algorithm in 
Kochenderfer chapter 6 on the hallway problem


######################################################
'''

from __future__ import division
from sys import path

path.append('../../src/');
from gaussianMixtures import GM, Gaussian 
from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 
from scipy.stats import norm; 
import time; 
from anytree import Node,RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree.iterators import PreOrderIter

class OnlineSolver():

	def __init__(self):
		modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass();
		self.N0 = 1; 
		self.Q0 = 100; 
		self.T = Node('',value = self.Q0,count=self.N0); 
		for a in range(0,self.model.acts):
			tmp = Node(self.T.name + str(a),parent = self.T,value=self.Q0,count=self.N0); 
		
		self.exploreParam = -1; 


	def beliefUpdate(self,bel,a,o):
		belBar = [0]*self.model.N; 
		belNew = [0]*self.model.N; 
		suma = 0; 
		for i in range(0,self.model.N):
			belBar[i] = sum(self.model.px[a][j][i]*bel[j] for j in range(0,self.model.N)); 
			belNew[i] = self.model.pz[o][i]*belBar[i];
			suma+=belNew[i];  
		#normalize
		for i in range(0,self.model.N):
			belNew[i] = belNew[i]/suma; 

		return belNew; 

	def MCTS(self,bel,d):
		h = self.T.name; 
		for a in range(0,self.model.acts):
			tmp = Node(self.T.name + str(a),parent = self.T,value=self.Q0,count=self.N0); 
		numLoops = 100; 

		for i in range(0,numLoops):
			s = np.random.choice([i for i in range(0,self.model.N)],p=bel);  
			self.simulate(s,h,d); 
		#RenderTreeGraph(self.T).to_picture('tree1.png'); 
		#print(RenderTree(self.T)); 
		QH = [0]*self.model.acts; 
		for a in range(0,self.model.acts):
			QH[a] = [node.value for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 

		act = np.argmax([QH[a] for a in range(0,self.model.acts)]);
		return [act,QH[act]]; 

	def simulate(self,s,h,d):

		if(d==0):
			return 0; 
		if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h)]) == 0):
			newRoot = [node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h[0:len(h)-1])][0];
			for a in range(0,self.model.acts):
				tmp = Node(h + str(a),parent = newRoot,value=self.Q0,count=self.N0); 
			tmp = Node(h,parent=newRoot,value = self.Q0,count=self.N0); 
			return self.getRolloutReward(s,d); 
		else:
			QH = [0]*self.model.acts; 
			NH = [0]*self.model.acts; 
			NodeH = [0]*self.model.acts; 
			for a in range(0,self.model.acts):
				QH[a] = [node.value for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 
				NH[a] = [node.count for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 
				NodeH[a] = [node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 

			aprime = np.argmax([QH[a] + self.exploreParam*np.sqrt(np.log(sum(NH)/NH[a])) for a in range(0,self.model.acts)]);  

			[sprime,o,r] = self.generate(s,aprime); 
			q = r + self.model.discount*self.simulate(sprime,h+str(aprime)+str(o),d-1); 
			NodeH[aprime].count += 1; 
			NodeH[aprime].value += (q-QH[a])/NH[a]; 
			return q; 

	def generate(self,s,a):
		sprime = np.random.choice([i for i in range(0,self.model.N)],p=self.model.px[a][s]);
		ztrial = [0]*len(self.model.pz); 
		for i in range(0,len(self.model.pz)):
			ztrial[i] = self.model.pz[i][sprime]; 
		z = ztrial.index(max(ztrial)); 
		reward = self.model.R[a][s]; 
		
		if(a == 0 and s > 13):
			reward = 10; 
		elif(a==1 and s<13):
			reward = 10; 
		elif(a == 2 and s==13):
			reward = 100;
		else:
			reward = -10; 
		

		return [sprime,z,reward]; 

	def getRolloutReward(self,s,d=1):
		reward = 0; 
		for i in range(0,d):
			a = np.random.randint(0,self.model.acts); 
			
			if(s < 13):
				a = 1; 
			elif(s>13):
				a = 0; 
			else:
				a = 2; 
			
			reward += self.model.discount*self.model.R[a][s]; 
			s = np.random.choice([i for i in range(0,self.model.N)],p=self.model.px[a][s]);
		return reward; 

	def normalize(self,a):
		suma = 0; 
		b=[0]*len(a); 
		for i in range(0,len(a)):
			suma+=a[i]
		for i in range(0,len(a)):
			b[i] = a[i]/suma; 
		return b; 


def testMCTS():
	# a = OnlineSolver(); 
	# b = [0.001 for i in range(0,a.model.N)]; 
	# b[13] = 1; 
	# b = a.normalize(b); 

	# action = a.MCTS(b,d=3); 


	a = OnlineSolver();
	allActs = [-1]*a.model.N; 
	allUtils = [-100]*a.model.N;
	for i in range(0,a.model.N):
		a = OnlineSolver(); 
		bel = [0.0001 for j in range(0,a.model.N)]; 
		bel[i] = 1; 
		bel = a.normalize(bel);
		[allActs[i],allUtils[i]] = a.MCTS(bel,5); 

	print(allActs); 

	fig,axarr = plt.subplots(2,sharex=True); 

	x = [i for i in range(0,a.model.N)]; 
 
	axarr[0].plot(x,allUtils,linewidth=5);
	axarr[0].set_title('MCTS Approximate Utility Function'); 

	grid = np.ones(shape=(1,a.model.N)); 

	
	axarr[1].scatter(-5,.5,c='k'); 
	axarr[1].scatter(-5,.5,c='r');
	axarr[1].scatter(-5,.5,c='y');

	for i in range(0,a.model.N):
		bel = [0]*a.model.N;  
		bel[i] = 1; 
		grid[0][i] = allActs[i];
	axarr[1].set_xlim([0,20]); 
	axarr[1].imshow(grid,extent=[0,a.model.N-1,0,1],cmap='inferno');
	axarr[1].legend(['Left','Right','Stay']); 
	axarr[1].set_title('Comparison to MDP policy implementation'); 

	plt.show();

def testMCTSSim():
	
	trails = 10; 
	trailLength = 100; 
	allReward = np.zeros(shape=(trails,trailLength)).tolist(); 

	for count in range(0,trails):
		if(trails == 1):
			fig,ax = plt.subplots();
		totalReward = 0; 

		a = OnlineSolver(); 
		b = np.random.rand(a.model.N).tolist(); 
		x = np.random.randint(0,a.model.N); 
		b[x] = 3; 
		b = a.normalize(b); 

		for step in range(0,trailLength):
			if(trails == 1):
				ax.cla(); 
				ax.plot(b,linewidth=4); 
				ax.scatter(x,.4,s=150,c='r'); 
				ax.set_ylim([0,.5]); 
				ax.set_title('POMCP Belief'); 
				plt.pause(0.1); 
			[act,u] = a.MCTS(b,3);  
			totalReward += a.model.R[act][x]; 
			x = np.random.choice([i for i in range(0,a.model.N)],p=a.model.px[act][x]);
			ztrial = [0]*len(a.model.pz); 
			for i in range(0,len(a.model.pz)):
				ztrial[i] = a.model.pz[i][x]; 
			z = ztrial.index(max(ztrial)); 
			b = a.beliefUpdate(b,act,z); 

			a.T = [node for node in PreOrderIter(a.T,filter_=lambda n: n.name==a.T.name+str(act)+str(z))][0];
			#RenderTreeGraph(a.T).to_picture('tree2.png');
			a.T.parent = None; 
			#print(a.T); 
			#RenderTreeGraph(a.T).to_picture('tree1.png');

			allReward[count][step] = totalReward;  

		print(allReward[count][-1]); 
 	
 	averageAllReward = [0]*trailLength; 
 	for i in range(0,trails):
 		for j in range(0,trailLength):
 			averageAllReward[j] += allReward[i][j]/trails; 
 	allSigma = [0]*trailLength; 

 	for i in range(0,trailLength):
 		suma = 0; 
 		for j in range(0,trails):
 			suma += (allReward[j][i] - averageAllReward[i])**2; 
 		allSigma[i] = np.sqrt(suma/trails); 
 	UpperBound = [0]*trailLength; 
 	LowerBound = [0]*trailLength; 

 	for i in range(0,trailLength):
 		UpperBound[i] = averageAllReward[i] + allSigma[i]; 
 		LowerBound[i] = averageAllReward[i] - allSigma[i]; 

 	x = [i for i in range(0,trailLength)]; 
 	plt.figure(); 
 	plt.plot(x,averageAllReward,'g'); 
 	plt.plot(x,UpperBound,'g--'); 
 	plt.plot(x,LowerBound,'g--'); 
 	plt.fill_between(x,LowerBound,UpperBound,color='g',alpha=0.25); 

 	plt.xlabel('Time Step'); 
	plt.ylabel('Accumlated Reward'); 
	plt.title('Average Accumulated Rewards over Time for: ' + str(trails) + ' simulations'); 

	plt.show(); 


if __name__ == "__main__":

	#testMCTS(); 
	testMCTSSim(); 

	# f = Node("f")
	# b = Node("b", parent=f)
	# a = Node("a", parent=b)
	# d = Node("d", parent=b)
	# c = Node("c", parent=d)
	# e = Node("e", parent=d)
	# g = Node("g", parent=f)
	# i = Node("i", parent=g)
	# h = Node("h", parent=i)

	# from anytree.iterators import PreOrderIter
	# print([node for node in PreOrderIter(f,filter_=lambda n: n.name=='h')])


	
	
