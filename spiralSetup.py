import numpy as np
import matplotlib.pyplot as plt
import copy
from math import sqrt





def aToxy(a):
	x = a/5; 
	y = a%5; 
	return [x,y]; 

def convertVectorToString(slices):
	slices = str(slices); 
	slices = slices.replace(']',''); 
	slices = slices.replace(',','');
	slices = slices.replace('[',''); 
	return slices;

def distance(x1,y1,x2,y2):
	a = (x1-x2)*(x1-x2); 
	b = (y1-y2)*(y1-y2); 
	return sqrt(a+b); 

def intitialize(spiralIn = True):
	#5 by 5 grid
	#deterministic movements
	#deterministic observations
	#reward for spiraling

	numStates = 25; 
	px = [[[0 for i in range(0,numStates)] for j in range(0,5)] for k in range(0,numStates)]; 

	#fill transition matrix
	for i in range(0,numStates):
		for j in range(0,numStates):
			[x1,y1] = aToxy(i); 
			[x2,y2] = aToxy(j); 

			if(distance(x1,y1,x2,y2) > 1):
				continue; 

			if(i == j):
				px[i][4][j] = 1; 
				continue; 


			if(x1-x2 == 1):
				px[i][0][j] = 1; 
			elif(x1-x2 == -1):
				px[i][1][j] = 1; 
			elif(y1-y2 == -1):
				px[i][2][j] = 1;
			elif(y1-y2 == 1):
				px[i][3][j] = 1; 
	
	#make sure that all actions lead to a state		
	for i in range(0,numStates):
		for j in range(0,5):
			if(sum(px[i][j]) == 0):
				px[i][j][i] = 1; 
	

	#fill observation matrix
	pz = [[0 for i in range(0,numStates)] for j in range(0,numStates)]; 

	for i in range(0,numStates):
		for j in range(0,numStates):
			if(i==j):
				pz[i][j] = 1; 


	#fill reward matrix
	r = [[-1 for i in range(0,5)] for j in range(0,numStates)]; 

	#ok so which states should you move which direction? 
	if(spiralIn == True):
		left = [17,18,21,22,23,24]; 
		right = [0,1,2,3,5,6,7,11]; 
		up = [9,14,19,4,8,13]; 
		down = [20,15,10,16]; 
		stay = [12]; 
	else:
		left = [12,8,7,6,4,3,2,1]; 
		right = [16,17,20,21,22,23]; 
		up = [11,5,10,15]; 
		down = [18,13,24,19,14,9]; 
		stay = [0]; 

	for i in range(0,numStates):
		if(i in left):
			r[i][0] = 100; 
		if(i in right):
			r[i][1] = 100; 
		if(i in up):
			r[i][2] = 100; 
		if(i in down):
			r[i][3] = 100; 
		if(i in stay):
			r[i][4] = 100; 

	return [px,pz,r]; 



def generateFile(In = True):

	print("Initializing Grid"); 
	if(In == True):
		SFile = open ("./SARSOPTests/GridSpiralIn25.pomdp","w"); 
	else:
		SFile = open ("./SARSOPTests/GridSpiralOut25.pomdp","w"); 
	num = 25; 
	[p,z,r] = intitialize(In); 

	print(p[24][1])

	print("Printing to File"); 


	print>>SFile,"discount: 0.95"
	print>>SFile,"values: reward"
	print>>SFile,"actions: left right up down wait"
	print>>SFile,"states: " + str(num)
	print>>SFile,"observations: " + str(num)


	print>>SFile,""
	print>>SFile,"T:wait"
	print>>SFile,"identity"
	print>>SFile,""
	print>>SFile,"T:left"


	#moving left, 0
	slices = [[p[y][0][x] for x in range(0,num)] for y in range(0,num)]; 
	for item in slices:
		tmp = convertVectorToString(item); 
		print>>SFile,tmp



	#moving right, 1
	print>>SFile,""
	print>>SFile,"T:right"

	slices = [[p[y][1][x] for x in range(0,num)] for y in range(0,num)]; 
	for item in slices:
		tmp = convertVectorToString(item); 
		print>>SFile,tmp


	#moving up, 2
	print>>SFile,""
	print>>SFile,"T:up"

	slices = [[p[y][2][x] for x in range(0,num)] for y in range(0,num)]; 
	for item in slices:
		tmp = convertVectorToString(item); 
		print>>SFile,tmp


	#moving down, 3
	print>>SFile,""
	print>>SFile,"T:down"

	slices = [[p[y][3][x] for x in range(0,num)] for y in range(0,num)]; 

	for item in slices:
		tmp = convertVectorToString(item);  
		print>>SFile,tmp

	#Observations
	print>>SFile,""
	print>>SFile,"O:*"
	for item in z:
		tmp = convertVectorToString(item); 
		print>>SFile,tmp


	for i in range(0,num):
		tmp = "R:left:" + str(i) + ":*:* " + str(r[i][0]); 
		print>>SFile,tmp; 
		tmp = "R:right:" + str(i) + ":*:* " + str(r[i][1]);
		print>>SFile,tmp; 
		tmp = "R:up:" + str(i) + ":*:* " + str(r[i][2]);
		print>>SFile,tmp; 
		tmp = "R:down:" + str(i) + ":*:* " + str(r[i][3]);
		print>>SFile,tmp; 
		tmp = "R:wait:" + str(i) + ":*:* " + str(r[i][4]);
		print>>SFile,tmp; 




def loadSarsopAlphas(fileName,l=25):
	#ok so we need a vector to hold these
	#but of what length? 
	#and how to store the action? 

	#so first grab the whole string, line by line? 
	#just have the last element in the al vector be the action
	num = sum(1 for line in open(fileName)); 
	als = [[0 for i in range(0,l+1)] for j in range(0,num-4)];
	lines = list(open(fileName)); 
	
	for i in range(0,len(lines)-4): 
		#find the first quotation, the next character is the action
		line = lines[i+3]; 
		mark1 = line.find("\""); 
		als[i][l] = int(line[mark1+1]);
		#find the first ">" and the second "<". Between the two are the coefficients
		line = line[mark1:]; 
		mark1 = line.find(">"); 
		mark2 = line.find("<"); 
		line = line[mark1+1:mark2]; 
		coeffs = line.split(); 
		for j in range(0,len(coeffs)):
			als[i][j] = float(coeffs[j]); 
	return als; 

def convertToSimpleAlphas(als,fileName):
	SFile = open (fileName,"w"); 
	for i in als:
		print>>SFile,convertVectorToString(i); 


#generateFile(False); 
als = loadSarsopAlphas("./SARSOPTests/GridSpiralOut25.policy",l = 25); 
#convertToSimpleAlphas(als,"./realAlphasSpiralOut25.txt"); 

'''
for i in als:
	print(i); 


#Use for confirmation

for i in range(0,25):
	tmp = [0 for j in range(0,25)]; 
	tmp[i] = 1; 
	
	bestAl = 0; 
	bestVal = -10000; 
 
	for j in range(0,len(als)):
		total = 0; 
		for k in range(0,25):
			total = total + tmp[k]*als[j][k]; 
		if(total > bestVal):
			bestVal = total; 
			bestAl = als[j][25]; 
	print(i,bestAl); 
'''
