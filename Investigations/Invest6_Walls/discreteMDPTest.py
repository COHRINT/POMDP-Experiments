from __future__ import division;
import numpy as np; 
import copy; 
import matplotlib.pyplot as plt;

'''
**************************
Test a discrete version of 
the D2WallsLocal Problem
**************************
'''


def buildReward(a,b):
	r = np.zeros(shape = (11,11)).tolist(); 
	r[9][1] = a; 
	for i in range(0,5):
		r[5][i] = b; 
	return r; 


def buildTransition():
	pt = np.zeros(shape = (11,11,5,11,11)); 
	for i in range(0,11):
		for j in range(0,11):
			for k in range(0,11):
				for l in range(0,11):
					if(i == k-1 and j==l):
						#left
						pt[i][j][0][k][l] = 1; 
					if(i == k+1 and j == l):
						#right
						pt[i][j][1][k][l] = 1; 
					if(i == k and j == l-1):
						#down
						pt[i][j][2][k][l] = 1;  
					if(i==k and j == l+1):
						#up
						pt[i][j][3][k][l] = 1; 
					if(i==k and j==l):
						pt[i][j][4][k][l] = 1; 
	return pt.tolist(); 

def listComp(a,b):
	for i in range(0,len(a)):
		for j in range(0,len(a[i])):
			if(a[i][j] != b[i][j]):
				return False; 
	return True; 

def ValueIteration(a,b):
	r = buildReward(a,b); 
	pt = buildTransition(); 
	discount = .9; 

	V = np.zeros(shape=(11,11)).tolist(); 
	for i in range(0,11):
		for j in range(0,11):
			V[i][j] = -100; 

	W = np.zeros(shape=(11,11)).tolist();
	counter = 0;  
	while(not listComp(V,W) and counter < 10000):
		counter += 1; 
		#print(counter); 
		W = copy.deepcopy(V); 
		for i in range(0,11):
			for j in range(0,11):
				maxim = -100000; 
				maxAct = 0;
				for a in range(0,5): 
					suma = 0; 
					for k in range(0,11):
						for l in range(0,11):
							suma+=V[k][l]*pt[i][j][a][k][l]; 
					suma += r[i][j]; 
					if(suma > maxim):
						maxim = suma;  
						maxAct = a; 
				V[i][j] = maxim*discount; 
	return V; 


x, y = np.mgrid[0:5.5:0.5, 0:5.5:0.5];




a = 100; 
allB = [-10,-100,-1000];

for b in allB:


	r = buildReward(a,b); 
	V = ValueIteration(a,b); 

	fig,axarr = plt.subplots(2); 
	levelsV = np.linspace(np.amin(V),np.amax(V)); 
	axarr[1].contourf(x,y,V,levelsV,cmap='viridis'); 
	axarr[1].set_title('Value Function'); 

	levelsR = np.linspace(np.amin(r),np.amax(r)); 
	axarr[0].contourf(x,y,r,levelsR,cmap='viridis'); 
	axarr[0].set_title('Reward Function'); 

	plt.suptitle('Success Reward: ' + str(a) + ', Wall Reward: ' + str(b));

	plt.pause(0.1);

plt.show(); 
