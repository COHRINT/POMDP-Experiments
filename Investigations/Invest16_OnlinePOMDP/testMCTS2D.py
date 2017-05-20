'''
######################################################

File: testMCTS2D.py
Author: Luke Burks
Date: April 2017

Implements the Monte Carlo Tree Search algorithm in 
Kochenderfer chapter 6 on the differencing problem


######################################################
'''

from __future__ import division
from sys import path

path.append('../../src/');
path.append('./models'); 
from gaussianMixtures import GM, Gaussian 
from copy import deepcopy;
import matplotlib.pyplot as plt; 
import numpy as np; 
from scipy.stats import norm; 
import time; 
from anytree import Node,RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree.iterators import PreOrderIter
from scipy.stats import multivariate_normal as mvn
import cProfile

class OnlineSolver():

	def __init__(self):
		modelModule = __import__('D2DiffsModel', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		self.model = modelClass();
		self.N0 = 1; 
		self.Q0 = 100; 
		self.T = Node('',value = self.Q0,count=self.N0); 
		for a in range(0,self.model.acts):
			if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==self.T.name+str(a))]) == 0):
				tmp = Node(self.T.name + str(a),parent = self.T,value=self.Q0,count=self.N0); 
				for o in range(0,self.model.obs):
					if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==(self.T.name+str(a)+str(o)))]) == 0):
						tmp2 = Node(self.T.name+str(a)+str(o),parent=tmp,value = self.Q0,count=self.N0);  
		
		self.exploreParam = -1; 


	def beliefUpdate(self,b,a,o,mod):
		btmp = GM(); 

		for obs in mod.pz[o].Gs:
			for bel in b.Gs:
				sj = np.matrix(bel.mean).T; 
				si = np.matrix(obs.mean).T; 
				delA = np.matrix(mod.delA[a]).T; 
				sigi = np.matrix(obs.var); 
				sigj = np.matrix(bel.var); 
				delAVar = np.matrix(mod.delAVar); 

				weight = obs.weight*bel.weight; 
				weight = weight*mvn.pdf((sj+delA).T.tolist()[0],si.T.tolist()[0],np.add(sigi,sigj,delAVar)); 
				var = (sigi.I + (sigj+delAVar).I).I; 
				mean = var*(sigi.I*si + (sigj+delAVar).I*(sj+delA)); 
				weight = weight.tolist(); 
				mean = mean.T.tolist()[0]; 
				var = var.tolist();
				 

				btmp.addG(Gaussian(mean,var,weight)); 
		btmp.normalizeWeights(); 
		btmp = btmp.kmeansCondensationN(1); 
		#btmp.condense(maxMix); 
		btmp.normalizeWeights();
		return btmp; 

	def MCTS(self,bel,d):
		h = self.T.name; 
		# for a in range(0,self.model.acts):
		# 	print(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==self.T.name+str(a))]))
		# 	if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==self.T.name+str(a))]) == 0):
		# 		tmp = Node(self.T.name + str(a),parent = self.T,value=self.Q0,count=self.N0); 
		# 		for o in range(0,self.model.obs):
		# 			if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==(self.T.name+str(a)+str(o)))]) == 0):
		# 				tmp2 = Node(self.T.name+str(a)+str(o),parent=tmp,value = self.Q0,count=self.N0); 
		#RenderTreeGraph(self.T).to_picture('tree2.png'); 
		numLoops = 100; 

		for i in range(0,numLoops):
			#s = np.random.choice([i for i in range(0,self.model.N)],p=bel); 
			s = bel.sample(1)[0];  
			self.simulate(s,h,d); 
		#RenderTreeGraph(self.T).to_picture('tree1.png'); 
		#print(RenderTree(self.T)); 
		QH = [0]*self.model.acts; 
		for a in range(0,self.model.acts):
			QH[a] = [node.value for node in PreOrderIter(self.T,filter_=lambda n: n.name==h+str(a))][0]; 

		#for pre, fill, node in RenderTree(self.T):
			#print("%s%s" % (pre, node.name))

		act = np.argmax([QH[a] for a in range(0,self.model.acts)]);
		return [act,QH[act]]; 

	def simulate(self,s,h,d):

		if(d==0):
			return 0; 
		if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h)]) == 0):
			newRoot = [node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h[0:len(h)-2])][0];
			newName = h[0:len(h)-2]; 
			for a in range(0,self.model.acts):
				if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==newName+str(a))]) == 0):
					tmp = Node(h + str(a),parent = newRoot,value=self.Q0,count=self.N0); 
					for o in range(0,self.model.acts):
						if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==newName+str(a)+str(o))]) == 0):
							tmp2 = Node(h+str(a)+str(o), parent = tmp,value = self.Q0,count=self.N0); 

			#tmp = Node(h,parent=newRoot,value = self.Q0,count=self.N0); 
			return self.getRolloutReward(s,d); 
		else:
			newRoot = [node for node in PreOrderIter(self.T,filter_=lambda n: n.name==h)][0];
			newName = h;
			for a in range(0,self.model.acts):
				if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==newName+str(a))]) == 0):
					tmp = Node(h + str(a),parent = newRoot,value=self.Q0,count=self.N0); 
					for o in range(0,self.model.acts):
						if(len([node for node in PreOrderIter(self.T,filter_=lambda n: n.name==newName+str(a)+str(o))]) == 0):
							tmp2 = Node(h+str(a)+str(o), parent = tmp,value = self.Q0,count=self.N0); 


			QH = [0]*self.model.acts; 
			NH = [0]*self.model.acts; 
			NodeH = [0]*self.model.acts;

			#print([node.name for node in PreOrderIter(self.T)])

			#RenderTreeGraph(self.T).to_picture('tree1.png'); 
			for a in range(0,self.model.acts):
				#print(h,a); 
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
		#sprime = np.random.choice([i for i in range(0,self.model.N)],p=self.model.px[a][s]);

		#tmpGM = GM((np.array(s) + np.array(self.model.delA)).T.tolist(),self.model.delAVar,1); 
		tmpGM = GM(); 
		tmpGM.addG(Gaussian((np.array(s) + np.array(self.model.delA[a])).tolist(),self.model.delAVar,1))

		sprime = tmpGM.sample(1)[0]; 
		ztrial = [0]*len(self.model.pz); 
		for i in range(0,len(self.model.pz)):
			ztrial[i] = self.model.pz[i].pointEval(sprime); 
		z = ztrial.index(max(ztrial)); 
		reward = self.model.r[a].pointEval(s); 
		
		'''
		if(a == 0 and s > 13):
			reward = 10; 
		elif(a==1 and s<13):
			reward = 10; 
		elif(a == 2 and s==13):
			reward = 100;
		else:
			reward = -10; 
		'''
		

		return [sprime,z,reward]; 

	def getRolloutReward(self,s,d=1):
		reward = 0; 
		for i in range(0,d):
			a = np.random.randint(0,self.model.acts); 
			
			'''
			if(s < 13):
				a = 1; 
			elif(s>13):
				a = 0; 
			else:
				a = 2; 
			'''

			reward += self.model.discount*self.model.r[a].pointEval(s); 
			#s = np.random.choice([i for i in range(0,self.model.N)],p=self.model.px[a][s]);
			tmpGM = GM(); 
			tmpGM.addG(Gaussian((np.array(s) + np.array(self.model.delA[a])).tolist(),self.model.delAVar,1))

			s = tmpGM.sample(1)[0]; 
		return reward; 

	def normalize(self,a):
		suma = 0; 
		b=[0]*len(a); 
		for i in range(0,len(a)):
			suma+=a[i]
		for i in range(0,len(a)):
			b[i] = a[i]/suma; 
		return b; 


def testMCTS2D():
	a = OnlineSolver(); 
	b = GM([2,0],[[0.15,0],[0,0.15]],1); 

	action = a.MCTS(b,d=2); 
	print(action); 




def testMCTSSim2D():
	
	trails = 10; 
	trailLength = 100; 
	allReward = np.zeros(shape=(trails,trailLength)).tolist(); 


	random = False; 

	for count in range(0,trails):
		'''
		if(trails == 1):
			fig,ax = plt.subplots();
		'''



		totalReward = 0; 

		a = OnlineSolver(); 
		x1 = np.random.randint(-5,5); 
		x2 = np.random.randint(-5,5); 
		x = [x1,x2]; 
		b = GM(); 
		b.addG(Gaussian(x,[[1,0],[0,1]],1)); 
		for step in range(0,trailLength):
			'''
			if(trails == 1):
				ax.cla(); 
				ax.plot(b,linewidth=4); 
				ax.scatter(x,.4,s=150,c='r'); 
				ax.set_ylim([0,.5]); 
				ax.set_title('POMCP Belief'); 
				plt.pause(0.1); 
			'''
			
			if(random):
				act = np.random.randint(0,5); 
			else:
				[act,u] = a.MCTS(b,2);  
			totalReward += a.model.r[act].pointEval(x); 
			#x = np.random.choice([i for i in range(0,a.model.N)],p=a.model.px[act][x]);
			tmpGM = GM(); 
			tmpGM.addG(Gaussian((np.array(x) + np.array(a.model.delA[act])).tolist(),a.model.delAVar,1))

			x = tmpGM.sample(1)[0]; 

			ztrial = [0]*len(a.model.pz); 
			for i in range(0,len(a.model.pz)):
				ztrial[i] = a.model.pz[i].pointEval(x); 
			z = ztrial.index(max(ztrial)); 
			b = a.beliefUpdate(b,act,z,a.model); 

			if(not random):
				#RenderTreeGraph(a.T).to_picture('tree2.png');
				a.T = [node for node in PreOrderIter(a.T,filter_=lambda n: n.name==a.T.name+str(act)+str(z))][0];
				
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

	testMCTS2D(); 
	#testMCTSSim2D(); 

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


	
	
