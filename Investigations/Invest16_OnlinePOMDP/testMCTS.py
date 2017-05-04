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
		self.Q0 = -100; 
		self.T = Node('',value = self.Q0,count=self.N0); 
		for a in range(0,self.model.acts):
			tmp = Node(self.T.name + str(a),parent = self.T,value=self.Q0,count=self.N0); 
		RenderTreeGraph(self.T).to_picture('tree3.png');
		


	def MCTS(self,bel,d):
		h = ''; 
		numLoops = 5; 

		for i in range(0,numLoops):
			s = np.random.choice([i for i in range(0,self.model.N)],p=bel);  
			self.simulate(s,h,d); 
		#RenderTreeGraph(self.T).to_picture('tree3.png'); 
		QH = [0]*self.model.acts; 
		for a in range(0,self.model.acts):
			QH[a] = [node.value for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 
		
		return np.argmax(QH[a] for a in range(0,self.model.acts)); 

	def simulate(self,s,h,d):
		if(d==0):
			return 0; 
		if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h)]) == 0):
			newRoot = [node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h[0:len(h)-2])][0];
			for a in range(0,self.model.acts):
				tmp = Node(newRoot.name + str(a),parent = newRoot,value=self.Q0,count=self.N0); 
			return self.getRolloutReward(s,d); 
		else:
			QH = [0]*self.model.acts; 
			NH = [0]*self.model.acts; 
			NodeH = [0]*self.model.acts; 
			for a in range(0,self.model.acts):
				QH[a] = [node.value for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 
				NH[a] = [node.count for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 
				NodeH[a] = [node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 

			aprime = np.argmax(QH[a] + np.sqrt(np.log(sum(NH)/NH[a])) for a in range(0,self.model.acts)); 
			[sprime,o,r] = self.generate(s,aprime); 
			q = r + self.model.discount*self.simulate(sprime,h+str(a)+str(o),d-1); 
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
		return [sprime,z,reward]; 

	def getRolloutReward(self,s,d=1):
		reward = 0; 
		for i in range(0,d):
			a = np.random.randint(0,self.model.acts); 
			reward += self.model.discount*self.model.R[a][s]; 
			s = np.random.choice([i for i in range(0,self.model.N)],p=self.model.px[a][s]); 
		return reward; 



def testMCTS():
	a = OnlineSolver(); 
	b = [0.001 for i in range(0,a.model.N)]; 
	b[4] = 1; 
	suma = sum(b); 
	for i in range(0,len(b)):
		b[i] = b[i]/suma; 

	action = a.MCTS(b,d=3); 
	print(action); 

if __name__ == "__main__":

	testMCTS(); 


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


	
	
