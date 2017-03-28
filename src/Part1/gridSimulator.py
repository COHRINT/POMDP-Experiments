import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import sys
import math
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib import cm
from scipy.stats import mvn
import pylab as pl
from random import random as rand
import matplotlib.animation as animation
import signal
import sys


#states are a 10 by 10 grid
#observations are also a 10 by 10 grid
#actions are go left, go right, go up, and go down, or wait
#reaching a goal state at (9,9) terminates the process
#Rewards are -1 for waiting, +100 for getting to the goal state, and -100 for hitting a wall
#walls will be implemented later
#observations will be based on a symetric two dimensional gaussian centered on the actual posistion
#the transition probabilities will be .85 for moving in the correct direction, and .15 spread among the rest
#0=left,1=right,2=up,3=down,4=wait


#TODO: Speed things up
#TODO: Double check the value iteration function

			
#problem specific implementation of the Bayes Filter from page 27 of Thrun
#takes about .005 seconds

controlCFlag = False; 

def gridBayesFilter(b,u,z,px,pz):
	if(u == 2):
		u = 3;
	elif(u==3):
		u = 2; 
	elif(u==1):
		u=0; 
	elif(u==0):
		u=1; 

	bNew = [0 for i in range(0,len(b))];
	belBar = [0 for i in range(0,len(b))];
	for i in range(0,len(b)):
		for j in range(0,len(b)):
			belBar[i] = belBar[i] + px[i][u][j]*b[j]; 
		bNew[i] = pz[i][z]*belBar[i]; 
	tmpsum = 0; 

	for i in range(0,len(b)):
		tmpsum = tmpsum + bNew[i]; 
	for i in range(0,len(b)):
		bNew[i] = bNew[i]/tmpsum; 
	return bNew; 
		


def getProbBelief(mean,var):
	if(var < .01):
		var = .01; 
	cov = [[var,0],[0,var]]; 
	b = [0 for i in range(0,100)];
	for i in range(0,100):
		b[i] = multivariate_normal.pdf([i%10,i/10],mean,cov); 
	b = normalize(b);
	return b; 

def extractBelief(bBar):
	#grab the mean
	mean = bBar[0]; 
	mean = [mean%10, mean/10]; 
	tmp = 2*(bBar[1] - (1+np.log(2*math.pi)));  
	tmp = math.exp(tmp); 
	var = math.pow(tmp,.5);   
	covariance = [[var,0],[0,var]];
	 
	b = getProbBelief(mean,var); 
	a = convertByF(b)[1]-bBar[1];
	
	sum = 0; 
	for i in range(0,len(b)):
		sum = sum + b[i]; 
	if(sum != 1):
		b[int(bBar[0])] = b[int(bBar[0])] + (1-sum); 

	return b; 
	#note to self, fix this
	#reimplemnt guassian, and inverse square both


def normForce(a):
	tmpsum = 0;
	b = [0 for k in range(0,len(a))];  
	for j in range(0,len(a)):
		tmpsum = tmpsum+a[j]; 
	for h in range(0,len(a)):
		b[h] = a[h]/tmpsum; 
	for l in range(0,len(a)):
		s = str(b[l])[0:4]; 
		b[l] = float(s); 
	normalize(b); 
	sum = 0; 
	for i in range(0,len(b)):
		sum = sum + b[i]; 
	if(sum != 1):
		b[b.index(max(b))] = b[b.index(max(b))] + (1-sum); 
	return b;


#takes negligible time
def normalize(a):
	tmpsum = 0;
	b = [0 for k in range(0,len(a))];  
	for j in range(0,len(a)):
		tmpsum = tmpsum+a[j]; 
	for h in range(0,len(a)):
		b[h] = a[h]/tmpsum; 
	return b; 
#takes about .0001 seconds
def convertByF(b):
	first = b.index(max(b)); 
	second = 0; 
	for k in range(len(b)):
		if(b[k] > 0.0001):
			second = second + b[k]*np.log(b[k]); 
	second = -second; 
	bBarPrime = [first,second]; 
	return bBarPrime;  



def printGrid(b):
	a = [[0 for i in range(0,10)] for j in range(0,10)]
	for i in range(0,100):
		a[i/10][i%10] = b[i]; 
	for i in range(0,10):
		print(a[i])






def getAMDPPolicy(px,pz,nValue = 10, perValue = 100, maxVal = 1000):


	#set the n value, which determines accuracy of learned values
	n = int(nValue); 
	
	#set percision of the probability, which in turn determines the number of possible entropies. 
	#remember to change the rounding number on the entropy backups when changing percision
	percision = float(perValue); 
	per = 100*percision+1;
	
	
	#preallocate space for the learned transition and reward tables
	P = [[[0 for i in range(0,100*int(percision))] for j in range(0,5)] for k in range(0,100*int(percision))];
	R = [[0 for u in range(0,5)] for j in range(0,100*int(percision))]; 
	
	
	allPossibleEntropy = [0 for i in range(0,int(perValue))]; 
	#maximum entropy is 4.7
	for j in range(0,len(allPossibleEntropy)):
		allPossibleEntropy[j] = (float(j)/percision)*4.7; 	
	
	print("Initialization Complete. Starting Learning Process"); 

	for i in range(0,100*int(percision)):		
	
		print("Learning Process Percent Completion: " + str(float(i)/(float(percision)))[0:5] + "%"); 
		#specify the current bbar
		bbar = [int(i/percision),allPossibleEntropy[i%int(percision)]]; 

		for u in range(0,5):

			for j in range(0,100*int(percision)):
				P[i][u][j] = 0; 
			R[i][u] = 0; 
			
			for N in range(0,n):
				#figure out b from bbar
				#use a function here
				b = extractBelief(bbar); 
				
				#sample a state, a single number 0:99, from b
					#weighted on the probability
				
				x = np.random.choice(np.linspace(0,99,100),p = b); 
				 			
			
				#from the motion model, get another state
					
				xPrime = 0; 
				if(u == 4):
					xPrime = x; 
				else: 
					slices = [px[k][u][int(x)] for k in range(0,100)];
					slices = normalize(slices); 
					xPrime = np.random.choice(np.linspace(0,99,100),p=slices);  	
				

				#sample an observation based on the new state
				z = int(np.random.choice(np.linspace(0,99,100),p=pz[int(xPrime)])); 
				

				#bayes filter
					#just make a function here
				bPrime = gridBayesFilter(b,u,z,px,pz);		
						 

				#convert b' into a new bbar using f(b)
				bBarPrime = convertByF(bPrime); 
				
				#Determine the state that you ended up with and increment the appropriate transition function by 1/n
				tmpFirst = bBarPrime[0]*int(percision); 
				nearest = 0; 
				best = 100000.0; 
				for k in range(0,len(allPossibleEntropy)):
					d = abs(bBarPrime[1]-allPossibleEntropy[k]); 
					if(d<best):
						best = d; 
						nearest = k; 
				tmpSecond = nearest; 
				
				#tmpSecond = int(min(allPossibleEntropy, key=lambda x:abs(x-bBarPrime[1])));	
				tmp = int(tmpFirst + tmpSecond); 
				P[i][u][tmp] = P[i][u][tmp] + 1.0/n; 
			

				#increment the Reward function by r(x,u)/n
				if(x == 99 and u == 4):
					R[i][u] = R[i][u] + 100.0/n; 
				else:
					R[i][u] = R[i][u] - 1.0/n; 


	'''
	#Uncomment for readability of output

	for i in range(0,int(percision)):
		for j in range(0,3):
			R[i][j] = int(R[i][j]); 
	 
	for i in range(0,int(percision)):
		for j in range(0,3):
			for k in range(0,int(percision)):
				P[i][j][k] = int(P[i][j][k]); 
	'''
	'''
	print("Learning Complete. Printing Learned Values.");
	PFile = open("./AMDPTests/PSmall.txt","w");
	for item in P:
		print>>PFile,item; 
	PFile.close();

	RFile = open ("./AMDPTests/RSmall.txt","w"); 
	for item in R:
		print>>RFile,item;
	RFile.close(); 
	print("Starting Value Iteration."); 
	'''


	print("Learning Complete. Starting Value Iteration.");
	#then do value iteration
	V = [-5000 for i in range(0,100*int(percision))];
	gamma = .95; 

	W = 0; 
	count = 0; 
	while(W!=V):
		W = copy.copy(V); 
		count = count+1; 
		if(count%10 == 0):
			print("Value Iteration Count: " + str(count)); 	
		if(count > maxVal):
			print("Suspected Value Function Error, computation limit overrun"); 
			break; 
		W = copy.copy(V); 
		for i in range(0,100*int(percision)):
			
			tmp0 = 0; 
			tmp1 = 0;
			tmp2 = 0; 
			tmp3 = 0; 
			tmp4 = 0;  
			for j in range(0,100*int(percision)):
				if(P[i][0][j] == 0 and P[i][1][j] == 0 and P[i][2][j] == 0 and P[i][3][j] == 0 and P[i][4][j] == 0):
					continue; 
				tmp0 = tmp0 + V[j]*P[i][0][j]; 
				tmp1 = tmp1 + V[j]*P[i][1][j]; 
				tmp2 = tmp2 + V[j]*P[i][2][j]; 
				tmp3 = tmp3 + V[j]*P[i][3][j]; 
				tmp4 = tmp4 + V[j]*P[i][4][j];
			tmp0 = tmp0 + R[i][0]; 
			tmp1 = tmp1 + R[i][1]; 
			tmp2 = tmp2 + R[i][2]; 
			tmp3 = tmp3 + R[i][3]; 
			tmp4 = tmp4 + R[i][4]; 
			

	
			V[i] = gamma*max(tmp0,tmp1,tmp2,tmp3,tmp4); 
		 
	
	print("Value Iteration Complete. Printing Results."); 	
	VFile = open("./AMDPTests/VNormalMedium.txt","w");
	for item in V:
		print>>VFile,item; 
	VFile.close();
	#print("Done. Files can be found in the AMDPTests folder in the current directory"); 
	#??simulate(V,P,R,allPossibleEntropy,percision); 
	
	

	print("Value iteration complete. Calculating Policy"); 
	#calculate policy for each bbar
	policy = [0 for i in range(0,len(V))];
	for i in range(0,len(V)):
		maxi = -10000.0; 
		maxU = 4; 
		for u in range(0,5):
			tmp = 0.0; 
			for j in range(0,len(V)):
				tmp = tmp + V[j]*P[i][u][j]; 
			tmp = tmp + R[i][u]; 
			if(maxi < tmp):
				maxi = tmp; 
				maxU = u; 
		policy[i] = maxU; 


	#output the policy to a file
	PolFile = open("./AMDPTests/PolicyNormalMedium.txt","w"); 
	for item in policy:
		print>>PolFile,item;
	PolFile.close(); 
	print("Done. Policy File can be found in the AMDPTests folder in the current directory.")


	'''
	#print(V)
	fig = plt.figure(); 
	ax = fig.add_subplot(111,projection='3d'); 
	gridV = [[ 0 for i in range(0,10)] for j in range(0,10)]; 
	
	for j in range(0,100):
		for i in range(0,int(percision)):
			ax.scatter(j/10,j%10,V[i*100+j]); 
	
	
	ax.set_zlabel('Value Function')
	plt.show(); 
	'''
	'''
	#plot the results
	Y = np.linspace(0,100,percision); 
	plt.plot(Y,V);  
	plt.ylabel('Value'); 
	plt.xlabel('Probability of the Tiger being behind Door 1'); 
	plt.title("Tiger AMDP Percision = " + str(percision) + ", n = " + str(n)); 
	plt.show(); 
	'''
	#then done




def MDPvalueIteration(P):
	#print("Starting Value Iteration");
	R = [-1 for i in range(0,100)]; 
	R[99] = 99; 
	V = [-120000 for i in range(0,100)]; 
	W = [0 for i in range(0,100)]; 
	gamma = .95; 
	while(V!=W):
		W = copy.copy(V); 
		for i in range(0,100):
			tmp0 = 0; 
			tmp1 = 0;
			tmp2 = 0; 
			tmp3 = 0; 
			tmp4 = 0;  
			for j in range(0,100):
				if(P[i][0][j] == 0 and P[i][1][j] == 0 and P[i][2][j] == 0 and P[i][3][j] == 0 and P[i][4][j] == 0):
					continue; 
				tmp0 = tmp0 + V[j]*P[i][0][j]; 
				tmp1 = tmp1 + V[j]*P[i][1][j]; 
				tmp2 = tmp2 + V[j]*P[i][2][j]; 
				tmp3 = tmp3 + V[j]*P[i][3][j]; 
				tmp4 = tmp4 + V[j]*P[i][4][j];
			tmp0 = tmp0 + R[i]; 
			tmp1 = tmp1 + R[i]; 
			tmp2 = tmp2 + R[i]; 
			tmp3 = tmp3 + R[i]; 
			tmp4 = tmp4 + R[i];
			V[i] = gamma*max(tmp0,tmp1,tmp2,tmp3,tmp4);
	return V; 




#takes about a second
def initializeGrid(cor = .85, zed = .5):


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
		z[i] = normalize(z[i]);


	r = [-1 for i in range(0,100)]; 
	r[99] = 99; 

	return p,z,r; 




def convertVectorToGrid(b):
	a = [[0 for i in range(0,10)] for j in range(0,10)]; 
	for i in range(0,100):
		a[i/10][i%10] = b[i]; 
	return a; 


def plotBelief(b):
	a = convertVectorToGrid(b);
	fig1=plt.figure(); 
	x = [i for i in range(0,10)]
	y = [i for i in range(0,10)] 
	X,Y = np.meshgrid(x,y); 
	Z = np.matrix(a); 
	ax = fig1.gca(projection='3d');
	ax.plot_surface(X,Y,Z,linewidth = 0.5,antialiased = True,cmap=cm.coolwarm,rstride = 1, cstride =1); 
	plt.show(); 

def loadAMDPPolicy(fileName,perValue):
	pol = [0 for i in range(0,100*perValue)]; 
	f = open(fileName,'r'); 
	x = 0; 
	for line in f:
		pol[x] = int(line); 
		x = x+1; 
	return pol; 



bet = 0; 
moves = 0; 
size = 0; 
num = 0; 
def update(data,line):
	global num; 
	mat.set_data(data);
	global moves; 

	if(len(moves)>=num):
		#a = ax.scatter(tmp%10,tmp/10,c='black',s = 50,marker='x'); 
		line.set_data(moves.astype(int)[:num]%10, moves.astype(int)[:num]/10) 
		line.axes.axis([-.5, 9.5, 9.5, -.5])
		num = num+1
	return mat; 

def data_gen():
	global bet; 
	global size; 
	size = len(bet)/100; 
	while size > 0:
		yield convertVectorToGrid(bet[:100]); 
		#print(convertVectorToGrid(bet[:9]))
		bet = bet[100:]; 
		size = size-1; 
		#print(bet); 



def loadAlphas(fileName):
	#ok so we need a vector to hold these
	#but of what length? 
	#and how to store the action? 

	#so first grab the whole string, line by line? 
	#just have the last element in the al vector be the action
	num = sum(1 for line in open(fileName)); 
	als = [[0 for i in range(0,101)] for j in range(0,num-4)];
	lines = list(open(fileName)); 
	
	for i in range(0,len(lines)-4): 
		#find the first quotation, the next character is the action
		line = lines[i+3]; 
		mark1 = line.find("\""); 
		als[i][100] = int(line[mark1+1]);
		#find the first ">" and the second "<". Between the two are the coefficients
		line = line[mark1:]; 
		mark1 = line.find(">"); 
		mark2 = line.find("<"); 
		line = line[mark1+1:mark2]; 
		coeffs = line.split(); 
		for j in range(0,len(coeffs)):
			als[i][j] = float(coeffs[j]); 
	return als; 
	

def PBVIloadAlphas(fileName):
	num = sum(1 for line in open(fileName)); 
	als = [[0 for i in range(0,101)] for j in range(0,num)];
	lines = list(open(fileName)); 
	with open(fileName) as f:
		lines = f.read().splitlines(); 
	linesNums = []; 
	for line in lines:
		tmp = []; 
		line = line.replace('[',''); 
		line = line.replace(']',''); 
		line = line.replace(',','');
		line = line.split(); 
		for i in range(0,len(line)):
			tmp.append(float(line[i])); 
		linesNums.append(tmp); 
	return linesNums; 


def grabPolicyFromAlphas(belief,als):
	most = -100000.0;
	mostInd = 0; 
	for i in range(0,len(als)):
		su = 0.0; 
		for j in range(0,len(belief)):
			su = su + belief[j]*als[i][j]; 
		#print(su);
		#print(i); 
		#print(""); 
		if(su > most):
			mostInd = i; 
			most = su;  

	#print(belief);
	#print(most); 
	#print(mostInd);  
	#print(""); 
	#print(als[mostInd])
	 
	return als[mostInd][100]; 

def getQMDPPolicy(belief,px,Q):
	arg = [0 for i in range(5)]; 
	for u in range(0,5):
		for i in range(0,100):
			arg[u] = arg[u] + belief[i]*Q[i][u]; 
	return arg.index(max(arg)); 

def getQ(px,value):
	Q = [[0 for i in range(0,5)] for j in range(0,100)]
	for xi in range(0,100):
		for u in range(0,4):
			Q[xi][u] = -1; 
			if(xi== 99):
				Q[xi][u] = 99; 
			for xj in range(0,100):
				Q[xi][u] = value[xj]*px[xi][u][xj]; 
	return Q; 

def PBVITao(b,a,o,px,pz):
	tao = [0.0 for i in range(0,100)];
	#i is s', j is s
	
	for i in range(0,100):
		for j in range(0,100):
			tao[i] = tao[i]+px[i][a][j]*b[j]; 
		tao[i] = pz[i][o]*tao[i]; 
	tao = normalize(tao); 
	return tao; 

def PBVIArgmaxAlphas(als,tao):

	best = als[0]; 
	bestVal = -1000; 
	for i in range(0,len(als)):
		tmpVal = 0; 
		for j in range(0,len(tao)):
			tmpVal = tmpVal+als[i][j]*tao[j]; 
		if(tmpVal > bestVal):
			bestVal = tmpVal; 
			best = als[i]; 
	return best;


def PBVIArgmaxActions(als,b):
	best = als[0]; 
	bestAction = 0;
	bestVal = -1000; 
	for a in range(0,5):
		tmpVal = 0;
		for i in range(len(b)):
			tmpVal = tmpVal + als[a][i]*b[i]; 
		if(tmpVal > bestVal):
			bestVal = tmpVal; 
			best = als[a]; 
			bestAction = a; 
	return [best,a]; 


def PBVIArgingStep1(als,b):
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


#TODO: Rewrite this based on Pineau
def PBVIBackup(B,Gamma,Actions,px,pz):
	#for all actions and observations
	'''
	als1 = [[0.0 for i in range(0,100)] for j in range(0,5)]; 
	als2 = [[0.0 for s in range(0,100)] for a in range(0,5)]; 
	for b in B: 
		for a in range(0,5):
			for o in range(0,100):
				als1[a][o] = PBVIArgmaxAlphas(Gamma,PBVITao(b,a,o,px,pz))
		for a in range(0,5):
			for s in range(0,100):
				for sprime in range(0,100):
					for o in range(0,100):
						als2[a][s] = als2[a][s]+px[s][a][sprime]*pz[sprime][o]*als1[a][o][sprime]
				als2[a][s] = .95*als2[a][s]; 
				if(s == 99):
					als2[a][s] = 99+als2[a][s]; 
				else:
					als2[a][s] = -1+als2[a][s]; 
		[alprime,acprime] = PBVIArgmaxActions(als2,b); 
		if((alprime in Gamma)!=True):
			Gamma.append(alprime);
			Actions.append(acprime); 
		if(controlCFlag == True):
				break;  

	return [Gamma,Actions]; 
	'''
	global controlCFlag; 

	als1 = [[[[0.0 for s in range(0,100)] for i in range(0,len(Gamma))] for a in range(0,5)] for z in range(0,100)]; 
	
	for a in range(0,5):
		for z in range(0,100):
			for i in range(0,len(Gamma)):
				for s in range(0,100):
					als1[z][a][i][s] = 0.0; 
					for sprime in range(0,100):
						als1[z][a][i][s] = als1[z][a][i][s]+ .95*px[s][a][sprime]*pz[sprime][z]*Gamma[i][sprime]; 
						if(controlCFlag == True):
							break;  

	GammaNew = [];
	ActionsNew = [];  
	for b in B:
		#Get reward
		#Get the alpha
		#Add them together
		#Then take the argmax

		bestAction = 0; 
		bestActionVal = -10000
		bestAlpha = []; 
		als2 = [[0.0 for s in range(0,100)]  for a in range(0,5)]; 
		for a in range(0,5):
			tmpVal = 0.0; 

			
			for z in range(0,100):
				tmp = PBVIArgingStep1(als1[z][a],b); 
				for sa in range(0,100):
					als2[a][sa] = als2[a][sa] + tmp[sa]; 

			for s in range(0,100):
				tmpR = -1.0; 
				if(s==99):
					tmpR = 99.0; 
				tmpR = tmpR*b[s];
				als2[a][s] = als2[a][s] + tmpR; 
				tmpVal = tmpVal + als2[a][s]; 



			if(bestActionVal < tmpVal):
				bestActionVal = tmpVal; 
				bestAction = a; 
				bestAlpha = als2[a]; 
		if((bestAlpha in GammaNew) == False):
			GammaNew.append(bestAlpha); 
			ActionsNew.append(bestAction); 
	return [GammaNew,ActionsNew]; 




def PBVIExpand(B,Gamma):
	Bnew = copy.copy(B); 

	for b in B:
		S = 100; 
		btmp = [0.0 for i in range(0,S)]; 
		for i in range(0,S):
			btmp[i] = random.random(); 
		btmp = sorted(btmp); 
		btmp = btmp[::-1]; 
		bnew = [0.0 for i in range(0,S)]

		for i in range(0,(S-1)):
			bnew[i] = btmp[i+1]-btmp[i]; 
		bnew[99] = btmp[99];
		bnew = normalize(bnew)
		Bnew.append(bnew); 
		if(controlCFlag == True):
				break; 
	return Bnew;


def PBVIMain(px,pz,N,T):
	#N is the number of expansions
	#T is the number of iterations in each expansion

	#Set up escape pod
	global controlCFlag; 

	#First we build the set of intial points. 
	#For now we'll use the uniform belief only, just to start with
	B = [[1.0 for i in range(0,100)] for j in range(0,1)]; 
	B[0] = normalize(B[0]);  
	#Now we intitialize Gamma zero to a low number Rmin/(1-gamma)
	gamma = .95; 
	Gamma = [[-1.0/(1-gamma) for i in range(0,100)] for j in range(0,1)]
	Actions = [1]; 
	for n in range(N): 
		for t in range(T):
			percent = (float(n*T)+float(t))/float(N*T); 
			date1 = math.exp(20.34444+3)-math.exp(3); 
			date2 = math.exp(20.34444*percent*percent*percent) - math.exp(3); 
			date = date1-date2
			#print("Percent Complete: " + str((float(n*T)+float(t))/float(N*T)*100) + "%"); 
			print("It is currently: " +str(date2) + " years ago."); 
			[Gamma,Actions] = PBVIBackup(B,Gamma,Actions,px,pz);  
			if(controlCFlag == True):
				break; 
		if(controlCFlag == True):
			break; 
		Bnew = PBVIExpand(B,Gamma); 
		B = Bnew; 
	print(Actions); 
	for i in range(0,len(Gamma)):
		Gamma[i].append(Actions[i]); 
	return Gamma; 





def simulatePOMDP(px,pz,r,sims,type,perValue = 100,verbose = False,walls =[]):
	#initialize the belief
	belief = [1.0 for i in range(0,100)]; 
	belief = normalize(belief); 

	#Set up the policy for the specified solver
	#type==0 -> AMDP
	#type==1 -> SARSOP
	#type==2 -> QMDP
	#type==3 -> PBVI

	if(type == 0):
		allPossibleEntropy = [0 for i in range(0,int(perValue))]; 
		#maximum entropy is 4.7
		for j in range(0,len(allPossibleEntropy)):
			allPossibleEntropy[j] = (float(j)/float(perValue))*4.7; 
		policy = loadAMDPPolicy("./AMDPTests/PolicyNormalMedium.txt",100); 
	elif(type == 1):
		policy = loadAlphas("./SARSOPTests/alphasShort1.txt");
	elif(type==2):
		MDPvalue = MDPvalueIteration(px); 
		Q = getQ(px,MDPvalue); 
	elif(type == 3):
		policy = PBVIloadAlphas("./PBVITests/alphasGood.txt")
	global bet; 
	global moves; 
	moves = np.array([]); 
	#moves = np.append(moves,0); 
	bet = np.array([]); 
	bet= np.append(bet,belief); 

	#Start at (0,0) with no reward
	if(type == 0):
		print("Simulating the policy: " + str(sims) + " times with AMDP.");
	elif(type==1):
		print("Simulating the policy: " + str(sims) + " times with SARSOP.");
	elif(type==2):
		print("Simulating the policy: " + str(sims) + " times with QMDP.");
	elif(type==3):
		print("Simulating the policy: " + str(sims) + " times with PBVI."); 

	allRewards = [0 for i in range(0,sims)]; 
	allSteps = [0 for i in range(0,sims)]; 
	#simulate the system.stop when you reach the goal
	for a in range(0,sims):
		belief = [1.0 for i in range(0,100)]; 
		#belief[0] = 2.0; 
		belief = normalize(belief); 
		x = 0; 
		totalReward = 0;
		steps = 0; 
		while(x!=99): 
			moves = np.append(moves,x);

			#print(x); 
			
			#get the policy
			#print(x);
			if(type == 0):
				#AMDP
				bbar = convertByF(belief);
				 
				tmpFirst = bbar[0]*int(perValue); 
				nearest = 0; 
				best = 100000.0; 
				for k in range(0,len(allPossibleEntropy)):
					d = abs(bbar[1]-allPossibleEntropy[k]); 
					if(d<best):
						best = d; 
						nearest = k; 
				tmpSecond = nearest; 
				index = int(tmpFirst+tmpSecond); 
				
				action = policy[index];
			elif(type == 1):
				#SARSOP
				action = grabPolicyFromAlphas(belief,policy); 
			elif(type == 2):
				#QMDP
				action = getQMDPPolicy(belief,px,Q); 
			elif(type == 3):
				action = int(grabPolicyFromAlphas(belief,policy));


			if(verbose):
				print(x); 
				print(action);
			

			x = int(np.random.choice(np.linspace(0,99,100), p = px[x][action]))
			if(verbose): 
				print(x); 
				print(""); 
			 
			#get a measurement
			z = int(np.random.choice(np.linspace(0,99,100),p=pz[int(x)]));

			#bayes filter
			belief = gridBayesFilter(belief,action,z,px,pz);
			bet= np.append(bet,belief);
			steps = steps + 1;   
			#get reward
			totalReward = totalReward+r[x]; 

		allRewards[a] = totalReward;
		#print(totalReward); 
		allSteps[a] = steps;  

	averageReward = float(sum(allRewards))/float(len(allRewards));
	averageSteps = float(sum(allSteps))/float(len(allSteps));  
	print("The average reward is: " + str(averageReward)); 
	print("The average number of steps is: " + str(averageSteps)); 
	print(""); 
 	

def sigint_handler(signum,frame):
	global controlCFlag; 
	controlCFlag = True; 
	print("Keyboard Interupt Detected. Stopping Execution.")


def getAllAlphaActions(als):
	actions = []; 
	for a in als:
		actions.append(int(a[100])); 
	return actions; 





def driver():
	#Takes about a second
	[p,z,r] = initializeGrid(.85,2);
	
	#getAMDPPolicy(p,z,10,100,2000); 
	'''
	for i in range(0,9): 
		slices = [z[i][k] for k in range(0,9)];
		slices = normForce(slices); 
		s = str(slices);
		s = s.replace(',',''); 
		s = s.replace('[',''); 
		s = s.replace(']',''); 
		print(s);  
	'''

	'''
	genNew = raw_input("Would you like to generate a new policy? (Y/N)"); 
	if(genNew == 'y' or genNew == 'Y')
:		if(len(sys.argv) > 1):
			main(p,z,sys.argv[1],sys.argv[2],sys.argv[3]); 
		else:
			main(p,z,20,100,200); 			
	'''

	#simulatePOMDP(px=p,pz=z,r=r,sims=100,type=2);
	
	for i in range(0,4):
		simulatePOMDP(px=p,pz=z,r=r,sims=100,type=i); 
	
	

signal.signal(signal.SIGINT,sigint_handler); 
#driver(); 





'''
[p,z,r] = initializeGrid(.85,2);
if(len(sys.argv) > 1):
	Gamma = PBVIMain(p,z,N=int(sys.argv[1]),T=int(sys.argv[2])); 
else:
	Gamma = PBVIMain(p,z,N=10,T=2);

print(Gamma)

AlphasFile = open("./PBVITests/alphas.txt","w"); 
for item in Gamma:
	print>>AlphasFile,item;
AlphasFile.close(); 
print("Done. The Alpha Vectors can be found in the PBVITests folder in the current directory.")
'''
'''
als = PBVIloadAlphas("./PBVITests/alphas.txt");
acts = getAllAlphaActions(als);
print(acts); 
'''
'''
[p,z,r] = initializeGrid(.85,2);
als = PBVIloadAlphas("./PBVITests/alphasGood.txt");
simulatePOMDP(p,z,r,sims=10,type = 3,verbose=False); 
'''

#Testing code for PBVIExpand
'''
b1 = [1.0 for i in range(0,100)]; 
b1 = normalize(b1); 
b2 = [1.0 for i in range(0,100)]; 
b2[25] = 2.0; 
b2 = normalize(b2); 
B = [b1,b2]; 
print(B); 
print("");
B = PBVIExpand(B,[]); 
B = PBVIExpand(B,[]); 
for b in B:
	print(len(b)); 
'''


#code for making .pomdp files
#TODO: formalize this into a function that can autogenerate the entire file
'''
[p,z] = initializeGrid(.85,3);
for i in range(0,100):
	slices = [z[i][k] for k in range(0,100)];
	slices = str(slices); 
	
	slices = slices.replace(']',''); 
	slices = slices.replace(',','');
	slices = slices.replace('[',''); 
	print(slices); 
'''

#Code to generate a gif
#TODO: put this in a function
'''
[p,z,r] = initializeGrid(.85,2);
als = PBVIloadAlphas("./PBVITests/alphasGood.txt");
simulatePOMDP(p,z,r,sims=1,type = 3,verbose=False); 
f = len(bet)/100; 
for i in range(0,len(bet)):
	bet[i] = bet[i]; 
fig,ax = plt.subplots(); 
mat = ax.matshow(convertVectorToGrid(bet[:100]),vmin = min(bet), vmax = max(bet)); 
#add walls
#walls at 5,15,25,50,51,52,66,67,57,47,76,75,74
#wallX = [5,5,5,0,1,2,6,7,7,7,6,5,4]; 
#wallY = [0,1,2,5,5,5,6,6,5,4,7,7,7]; 
#ax.scatter(wallX,wallY,color='r',marker='s',s=750); 

x = np.linspace(0, 10, 100)
y = np.sin(x)

line, = ax.plot(x, y, c='w',lw = 5)

plt.colorbar(mat); 
ani = animation.FuncAnimation(fig, update,frames=data_gen, interval=500, save_count=100, blit=False, fargs = [line]); 
#plt.show();

ani.save('animation.gif', writer='imagemagick', fps=2)
'''
