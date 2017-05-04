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


class Node():
	def __init__(self,parent=None,value = 0):
		self.parent = parent;  
		self.children = []; 
		self.value = value; 

	def findNode(self,n):
		if(n in self.children):
			return self.children()




	def addChild(self,child):
		self.children.append(child); 
		child.parent = self; 






class OnlineSolver():


	def __init__(self):
		modelModule = __import__('hallwayProblemSpec', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass();
		self.T = Node(1); 
		self.N0 = 0; 
		self.Q0 = -100; 

		self.T.addChild(Node(2)); 
		C = Node(3); 
		C.addChild(Node(4)); 
		self.T.addChild(C); 
		result = []; 
		self.T.getAllChildren(result); 
		print(result)



	def MCTS(self,bel,d):
		h = []; 
		numLoops = 5; 

		for i in range(0,numLoops):
			s = np.random.choice([i for i in range(0,self.model.N)],p=bel); 
			self.simulate(s,h,d); 
		return np.argmax(self.Q(h,a) for a in range(0,self.model.acts)); 

	def simulate(self,s,h,d):
		if(d==0):
			return 0; 
		if(h not in T):
			for a in range(0,self.model.acts):
				#add to tree
				j = 0; 
			return 0; 
		else:
			aprime = np.argmax(Q(h,a) + np.sqrt(np.log(sum(N(h)))/N(h,a)) for a in range(0,self.model.acts)); 
			[sprime,o,r] = generate(s,aprime); 
			q = r + self.model.discount*simulate(sprime,hao,d-1); 
			N(h,a) += 1; 
			Q(h,a) += (q-Q(h,a))/N(h,a); 
			return q; 



def testMCTS():
	a = OnlineSolver(); 


if __name__ == "__main__":

	testMCTS(); 
