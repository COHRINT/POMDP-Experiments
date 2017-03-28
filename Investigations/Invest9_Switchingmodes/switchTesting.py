from __future__ import division;
import numpy as np; 
import copy; 
import matplotlib.pyplot as plt;
from sys import path
path.append('/home/luke/Documents/POMDP/src');

from gaussianMixtures import Gaussian
from gaussianMixtures import GM
from scipy.stats import multivariate_normal as mvn

'''
***********************************************
File: switchTesting.py
Author: Luke Burks
Date: January 2017

Testing the swtiching mode extension of CPOMDP
suggested in Brunskill 2010

Simple 1D Problem?

Do the corridor problem, -20,20 with a door
somewhere. 

3 Modes: left wall, right wall, no wall
3 Actions: left, right, declare victory
1 Blank observation? 
Reward at the door
***********************************************
'''




def buildTransitions():
	delA = [[0 for i in range(0,3)] for j in range(0,3)]
	modes = [0]*3; 

	#Mode 0: left wall
	modes[0] = GM(); 
	modes[0].addNewG(-18,2,3); 
	#Act 0: left
	delA[0][0] = 0; 

	#Act 1: right
	delA[0][1] = 1; 

	#Act 2: stay
	delA[0][2] = 0; 



	#Mode 1: Right Wall
	modes[1] = GM(); 
	modes[1].addNewG(18,2,3); 
	#Act 0: left
	delA[1][0] = -1; 

	#Act 1: right
	delA[1][1] = 0; 

	#Act 2: stay
	delA[1][2] = 0; 


	#Mode 2: Corridor
	modes[2] = GM(); 
	for i in range(-16,17):
		modes[2].addNewG(i,2,1); 
	#Act 0: left
	delA[2][0] = -1; 

	#Act 1: right
	delA[2][1] = 1; 

	#Act 2: stay
	delA[2][2] = 0;


	delAVar = 0.01; 

	
	#Test and Verify
	[a0,b0] = modes[0].plot(vis=False,low=-20,high=20); 
	[a1,b1] = modes[1].plot(vis=False,low=-20,high=20); 
	[a2,b2] = modes[2].plot(vis=False,low=-20,high=20); 

	suma = [0]*len(b0); 
	for i in range(0,len(b0)):
		suma[i]+=b0[i]; 
		suma[i]+=b1[i]; 
		suma[i]+=b2[i]; 

	lwidth = 10; 
	plt.plot(a0,b0,linewidth = lwidth); 
	plt.plot(a1,b1,linewidth = lwidth); 
	plt.plot(a2,b2,linewidth = lwidth); 
	#plt.plot(a0,suma);  
	plt.legend(['Left Wall','Right Wall','Open Space']); 
	plt.title('Softmax Transition Modes in 1D')
	plt.show(); 
	

	return delA,modes,delAVar;

def buildObs():
	'''
	pz = [GM()];
	for i in range(-5,6): 
		pz[0].addNewG(i*4,4,4); 
	#pz[0].plot(low=-20,high = 20); 
	return pz; 
	'''

	pz = [0]*3;

	for i in range(0,3):
		pz[i] = GM(); 

	for i in range(-5,6):
		if(i*4 <= 0):
			pz[0].addNewG(i*4,4,.2); 
		elif(i*4 >= 8):
			pz[1].addNewG(i*4,4,.2); 
		else:
			pz[2].addNewG(i*4,4,.2); 

	#for i in range(0,3):
		#pz[i].plot(low=-20,high=20); 


	return pz; 


def buildRewards():
	r = [0]*3; 
	for i in range(0,3):
		r[i] = GM(); 

	#door location
	dloc = 4; 

	#left
	for i in range(-5,6):
		r[0].addNewG(i*4,4,.2); 
		r[1].addNewG(i*4,4,.2); 
		#a = 0; 
	r[2].addNewG(dloc,0.25,15); 

	'''
	for i in range(0,3):
		r[i].plot(low=-20,high=20); 
	'''

	return r; 


def beliefUpdate(modes,delA,delAVar,pz,bels,a,o,cond = -1):
	
	#Initialize
	btmp = GM(); 

	for d in bels.Gs:
		for h in modes:
			for f in h.Gs:
				for l in pz[o].Gs:
					C1 = 1/(1/f.var + 1/d.var);
					c1 = C1*((1/f.var)*f.mean + (1/d.var)*d.mean); 

					C2 = C1 + delAVar; 
					c2 = c1+delA[modes.index(h)][a]; 

					weight = d.weight*f.weight*l.weight*mvn.pdf(l.mean,c2,l.var+C2); 

					var = 1/((1/l.var)+(1/C2)); 
					mean = var*((1/l.var)*l.mean + (1/C2)*c2); 

					g = Gaussian(mean,var,weight); 
					btmp.addG(g); 


	btmp.normalizeWeights(); 


	if(cond != -1):
		btmp = btmp.kmeansCondensationN(k=cond,lowInit = -20,highInit=20); 
		#btmp.condense(cond); 

	for g in btmp:
		while(isinstance(g.var,list)):
			g.var = g.var[0]; 

	btmp.display(); 

	return btmp; 

def continuousDot(a,b):
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

def backup(als,modes,delA,delAVar,pz,r,maxMix,b):
	
	newAls = [[[0 for i in range(0,len(pz))] for j in range(0,len(delA[0]))] for k in range(0,len(als))]; 

	for i in range(0,len(als)):
		for j in range(0,len(delA[0])):
			for k in range(0,len(pz)):
				newAls[i][j][k] = GM(); 
				
				
				for h in modes:
					tmpGM = als[i].GMProduct(pz[j]); 
					mean = tmpGM.getMeans(); 
					for l in range(0,len(mean)):
						mean[l][0] -= delA[modes.index(h)][j]; 
						mean[l] = mean[l][0]; 
					var = tmpGM.getVars(); 
					for l in range(0,len(var)):
						var[l][0][0] += delAVar; 
						var[l] = var[l][0][0];
					weights = tmpGM.getWeights(); 
					tmpGM2 = GM(); 
					for l in range(0,len(mean)):
						tmpGM2.addG(Gaussian(mean[l],var[l],weights[l]))
					#tmpGM2 = GM(mean,var,tmpGM.getWeights()); 

					newAls[i][j][k].addGM(tmpGM2.GMProduct(h)); 

	bestVal = -10000000000; 
	bestAct= 0; 
	bestGM = []; 

	for a in range(0,len(delA[0])):
		suma = GM(); 
		for o in range(0,len(pz)):
			suma.addGM(newAls[np.argmax([continuousDot(newAls[j][a][o],b) for j in range(0,len(newAls))])][a][o]); 
		suma.scalerMultiply(0.9); 
		suma.addGM(r[a]); 

		for g in suma.Gs:
			if(isinstance(g.mean,list)):
				g.mean = g.mean[0]; 
				g.var = g.var[0][0]; 


		suma = suma.kmeansCondensationN(k=maxMix,lowInit = -20,highInit=20); 

		tmp = continuousDot(suma,b);
		#print(a,tmp); 
		if(tmp > bestVal):
			bestAct = a; 
			bestGM = copy.deepcopy(suma); 
			bestVal = tmp; 

	bestGM.action = bestAct; 

	return bestGM; 


def solve(B,modes,delA,delAVar,pz,r,loops = 100):

	Gamma = copy.deepcopy(r[2]); 
	Gamma.scalerMultiply(1/.1); 
	Gamma = [Gamma]; 

	maxMix = 10; 

	for counter in range(0,loops):
		print("Iteration: " + str(counter+1));
		bestAlphas = [GM()]*len(B); 
		Value = [0]*len(B);
		GammaNew = []; 

		for b in B:
			bestAlphas[B.index(b)] = Gamma[np.argmax([continuousDot(Gamma[j],b) for j in range(0,len(Gamma))])];
			Value[B.index(b)] = continuousDot(bestAlphas[B.index(b)],b);

		for b in B:
			al = backup(Gamma,modes,delA,delAVar,pz,r,maxMix,b)
			tmpVal = continuousDot(al,b); 
			if(tmpVal > Value[B.index(b)]):
				GammaNew.append(al); 
		Gamma = copy.deepcopy(GammaNew);


		print("Number of Alphas: " + str(len(GammaNew))); 
		av = 0; 
		for i in range(0,len(GammaNew)):
			av += GammaNew[i].size; 
		av = av/len(GammaNew);  
		print("Average number of mixands: " + str(av));


		print("Actions: " + str([GammaNew[i].action for i in range(0,len(GammaNew))])); 
		print("");

	return Gamma; 

def getAction(b,Gamma):
	act = Gamma[np.argmax([continuousDot(j,b) for j in Gamma])].action;
	return act; 


#Build models
[delA,modes,delAVar] = buildTransitions(); 
pz = buildObs();
r = buildRewards(); 
 

#Make belief
b = GM(); 
b.addNewG(-15,1,1); 

act = 2; 
obs = 2; 
maxMix = 10; 


B = []; 
'''
B.append(b); 
b = GM(); 
b.addNewG(4,1,1); 
B.append(b); 
b = GM(); 
b.addNewG(15,1,1); 
B.append(b); 
'''
'''
for i in range(-5,6):
	b = GM(); 
	b.addG(Gaussian(i*4,0.1,1)); 
	#b.addNewG(i*4,1,1); 
	B.append(b); 
'''
b = GM(); 
b.addG(Gaussian(-17,0.01,1)); 
B.append(b); 
b = GM(); 
b.addG(Gaussian(17,0.01,1)); 
B.append(b);
b = GM(); 
b.addG(Gaussian(4,0.01,1)); 
B.append(b);
b = GM(); 
b.addG(Gaussian(3,0.01,1)); 
B.append(b);
b = GM(); 
b.addG(Gaussian(5,0.01,1)); 
B.append(b);
b = GM(); 
b.addG(Gaussian(0,0.01,1)); 
B.append(b);


'''
Gamma = solve(B,modes,delA,delAVar,pz,r,loops=2); 

for i in Gamma:
	#i.normalizeWeights(); 
	i.display(); 
	i.plot(low=-20,high=20); 

f = open('switchPolicy1D_2.npy',"w"); 
np.save(f,Gamma); 
f.close();
'''


Gamma = np.load('switchPolicy1D_1.npy');


'''
for i in Gamma:
	print(i.action); 
	i.plot(low=-20,high=20); 
'''


xs = [i for i in range(-20,21)]; 

gs = [0]*len(xs); 
for i in range(0,len(xs)):
	gs[i] = GM(); 
	gs[i].addG(Gaussian(i-20,5,1));  
acts = [-1]*len(xs); 
for i in range(0,len(xs)):
	acts[i] = getAction(gs[i],Gamma); 

print(acts); 




'''
al1 = [GM()]; 
al1 = copy.deepcopy(r[2]); 
al1.scalerMultiply(1/(.1)); 
al1 = [al1]; 

al2 = backup(al1,modes,delA,delAVar,pz,r,maxMix,b)

al2.plot(low=-20,high=20); 
'''

'''
b.plot(low = -20,high=20); 
b = beliefUpdate(modes,delA,delAVar,pz,b,act,obs,5); 
b.plot(low = -20,high=20); 
'''

