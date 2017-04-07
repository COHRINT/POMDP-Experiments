from __future__ import division
from sys import path

path.append('../../src/');
from gaussianMixtures import GM, Gaussian 
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import numpy as np
import copy
import matplotlib.pyplot as plt
#Exploring Monte Carlo POMDP methods

#Step 1: Build a working particle filter



#Algorithm in Probabilistic Robotics by Thrun, page 98
def particleFilter(Xprev,a,o,pz):
	Xcur = []; 

	delA = [-1,1,0]; 
	delAVar = 0.5;
	allW = []; 
	allxcur = []; 
	for xprev in Xprev:
		xcur = np.random.normal(xprev+delA[a],delAVar,size =1)[0].tolist(); 
		
		w = pz[o].pointEval(xcur); 
 
		allW.append(w);
		allxcur.append(xcur); 

	#normalize weights for kicks
	suma = 0; 
	for w in allW:
		suma+=w; 
	for i in range(0,len(allW)):
		allW[i] = allW[i]/suma; 

	for m in range(0,len(Xprev)):
		c = np.random.choice(allxcur,p=allW); 
		Xcur.append(copy.deepcopy(c)); 
	return Xcur; 


def beliefUpdate(b,a,o,pz):
		btmp = GM(); 

		adelA = [-1,1,0]; 
		adelAVar = 0.5;

		for obs in pz[o].Gs:
			for bel in b.Gs:
				sj = np.matrix(bel.mean).T; 
				si = np.matrix(obs.mean).T; 
				delA = np.matrix(adelA[a]).T; 
				sigi = np.matrix(obs.var); 
				sigj = np.matrix(bel.var); 
				delAVar = np.matrix(adelAVar); 

				weight = obs.weight*bel.weight; 
				weight = weight*mvn.pdf((sj+delA).T.tolist()[0],si.T.tolist()[0],np.add(sigi,sigj,delAVar)); 
				var = (sigi.I + (sigj+delAVar).I).I; 
				mean = var*(sigi.I*si + (sigj+delAVar).I*(sj+delA)); 
				weight = weight.tolist(); 
				mean = mean.T.tolist()[0]; 
				var = var.tolist();
				 

				btmp.addG(Gaussian(mean,var,weight)); 


		btmp.normalizeWeights(); 
		btmp.condense(1); 
		btmp.normalizeWeights(); 


		return btmp; 


#Defined (sloppily) as the average distance of every 1 particle from every 2 particle
def particleSetDistance(X1,X2):
	
	suma = 0; 
	for i in range(0,len(X1)):
		for j in range(0,len(X2)):
			suma += abs(X1[i] - X2[j]); 
	aveDist = suma/(len(X1)*len(X2)); 

	return aveDist; 


def shepardsInterpolation(V,XPRIME,al = .1,retDist=False):
	
	dists = [0]*len(V); 

	#Upper triangular
	for i in range(0,len(V)):
			dists[i] = particleSetDistance(V[i][0],XPRIME); 

	#find points within a threshold
	distSort = sorted(set(dists));

	if(retDist == False):
		thres = 1; 
		cut = 1; 
		for i in range(0,len(distSort)):
			if(distSort[i]>1):
				cut = i; 
				break; 
		distSort = distSort[:cut];  

	eta = 0; 
	suma = 0; 
	for i in range(0,len(distSort)):
		eta += 1/dists[dists.index(distSort[i])]; 
		suma += (1/dists[dists.index(distSort[i])])*V[dists.index(distSort[i])][1]; 

	if(eta>0):	
		eta = 1/eta; 
	ans = eta*suma; 

	if(retDist):
		return [distSort,dists,eta]
	else:
		return ans; 


#Algorithm in Probabilistic Robotics by Thrun, page 560
def MCPOMDP(b0,M=100,iterations=100):
	V=[]; 
	delA = [-1,1,0]; 
	delAVar = 0.1;
	R = GM(4,0.25,1); 
	simLoopsN = 10; 
	gamma = .9; 

	#learning param
	alpha = 0.1; 

	pz = [GM(),GM(),GM()]; 
	for i in range(-10,4):
		pz[0].addG(Gaussian(i,1,1)); 
	pz[1].addG(Gaussian(4,1,1)); 
	for i in range(5,10):
		pz[2].addG(Gaussian(i,1,1));


	#until convergence or time
	for count in range(0,iterations):
		#sample x from b
		#[mean,var] = (b0.getMeans()[0],b0.getVars()[0])
		#x = np.random.normal(mean,var); 

		#sample particle set from b 
		X = b0.sample(M); 
		
		#for each episode?
		for part in X:
			Q = [0]*len(delA); 
			#for each action
			for a in range(0,len(delA)):
				#Simulate possible new beliefs
				for n in range(0,simLoopsN):
					x = part; 
					xprime = np.random.normal(x+delA[a],delAVar,size =1)[0].tolist(); 
					ztrial = [0]*len(pz); 
					for i in range(0,len(pz)):
						ztrial[i] = pz[i].pointEval(xprime); 
					z = ztrial.index(max(ztrial)); 
					XPRIME = particleFilter(X,a,z,pz); 
					Q[a] = Q[a] + (1/simLoopsN)*gamma*(R.pointEval(xprime) + shepardsInterpolation(V,XPRIME)); 

			[distSort,dists,eta] = shepardsInterpolation(V,X,retDist=True); 
			#update used value entries
			for i in range(0,len(distSort)):
				tmpVal = V[dists.index(distSort[i])][1] + alpha*eta*(1/dists[dists.index(distSort[i])])*(max(Q)-V[dists.index(distSort[i])][1]);
				#V[dists.index(distSort[i])] = [V[dists.index(distSort[i])][0],tmpVal,Q.index(max(Q))]; 
				V[dists.index(distSort[i])] = [V[dists.index(distSort[i])][0],tmpVal,V[dists.index(distSort[i])][2]]; 

			act = Q.index(max(Q)); 
			V.append([X,max(Q),act]); 



			xprime = np.random.normal(x+delA[act],delAVar,size =1)[0].tolist(); 
			ztrial = [0]*len(pz); 
			for i in range(0,len(pz)):
				ztrial[i] = pz[i].pointEval(xprime); 
			z = ztrial.index(max(ztrial));
			Xprime = particleFilter(X,act,z,pz); 
			x = xprime; 
			X =	copy.deepcopy(Xprime); 

	return V




def testParticleFilter():
	pz = [GM(),GM(),GM()]; 
	for i in range(-5,0):
		pz[0].addG(Gaussian(i,1,1)); 
	pz[1].addG(Gaussian(0,1,1)); 
	for i in range(0,5):
		pz[2].addG(Gaussian(i,1,1)); 
	initAct= GM(0,0.5,1); 
	actSeq = [0,0,0,0,1,1,1,1,1,1,2]; 
	obsSeq = [1,0,0,0,0,0,1,1,2,2,2]; 
	numParticles = 100; 

	initPart = []; 
	for i in range(0,numParticles):
		initPart.append(np.random.normal(0,0.5)); 

	seqPart = []; 
	seqPart.append([initPart]); 
	seqAct = []; 
	seqAct.append(initAct); 
	for i in range(0,len(actSeq)):
		
		tmp = particleFilter(seqPart[i][0],actSeq[i],obsSeq[i],pz); 
		seqPart.append([tmp]); 
		
		tmp = beliefUpdate(seqAct[i],actSeq[i],obsSeq[i],pz)

		seqAct.append(tmp); 

	allSigmasAct = []; 
	allSigmasPart = []; 
	for i in range(0,len(seqPart)):
		mean = 0; 
		var = 0; 
		for j in range(0,len(seqPart[i])):
			mean+=seqPart[i][0][j]/len(seqPart[i]); 
		for j in range(0,len(seqPart[i][0])):
			var += (seqPart[i][0][j]-mean)*(seqPart[i][0][j]-mean)/len(seqPart[i][0]); 
		allSigmasPart.append(np.sqrt(var)); 
		allSigmasAct.append(np.sqrt(seqAct[i].getVars()[0])); 

	for i in range(0,len(allSigmasAct)):
		while isinstance(allSigmasAct[i],list) or isinstance(allSigmasAct[i],np.ndarray):
			allSigmasAct[i] =allSigmasAct[i].tolist()[0][0]
	diffs = []; 
	ratios = [];
	averageDiff=0; 
	averageRatio = 0; 
	for i in range(0,len(allSigmasAct)):
		diffs.append(allSigmasPart[i]-allSigmasAct[i]); 
		ratios.append(allSigmasPart[i]/allSigmasAct[i]); 
		averageDiff += diffs[i]/len(allSigmasAct); 
		averageRatio += ratios[i]/len(allSigmasAct); 

	fig,axarr = plt.subplots(len(actSeq),1); 
	for i in range(0,len(actSeq)):
		[x,c] = seqAct[i].plot(low=-5,high=5,vis=False); 
		axarr[i].plot(x,c,color='r'); 
		axarr[i].hist(seqPart[i],normed=1,bins=10); 
		axarr[i].set_xlim([-5,5]);
		#print(str(i) + ' Part Sig:' + str(allSigmasPart[i]) + '  Act Sig:' + str(allSigmasAct[i]) + ' Diff:' + str(allSigmasPart[i]-allSigmasAct[i]));

	print('Average Sigma Difference: ' + str(averageDiff)); 
	print('Average Sigma Ratio: ' + str(averageRatio)); 
	plt.show(); 


def displayPolicy(V):
	for v in V:
		plt.hist(v[0],normed=1,bins=10);
		print(v[2]); 
		plt.pause(0.1);  

def testMCPOMDP():
	b = GM(4,1,1);
	b.addG(Gaussian(-1,1,1)); 
	b.addG(Gaussian(10,1,1));  
	V= MCPOMDP(b,20,2); 
	
	displayPolicy(V); 

	numParticles = 10; 
	testX1 = []; 
	testX2 = []; 
	testX3 = []; 
	for i in range(0,numParticles):
		testX1.append(np.random.normal(-2,0.5)); 
		testX2.append(np.random.normal(4,0.5)); 
		testX3.append(np.random.normal(10,0.5)); 


	#test 1
	[distSort1,dists1,eta1] = shepardsInterpolation(V,testX1,retDist=True); 
	act1 = V[dists1.index(distSort1[0])][2]; 

	#test 2
	[distSort2,dists2,eta2] = shepardsInterpolation(V,testX2,retDist=True); 
	act2 = V[dists2.index(distSort2[0])][2]; 

	#test3
	[distSort3,dists3,eta3] = shepardsInterpolation(V,testX3,retDist=True); 
	act3 = V[dists3.index(distSort3[0])][2]; 

	print(act1,act2,act3); 

if __name__ == "__main__":

	#testParticleFilter(); 
	testMCPOMDP(); 


