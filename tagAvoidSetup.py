import numpy as np
import matplotlib.pyplot as plt
import copy
from math import sqrt






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

def normalize(a):
	Suma = sum(a); 
	for i in range(0,len(a)):
		a[i] = a[i]/Suma; 
	return a; 

def aToxy(a):
	x1 = int(a/1000);
	y1 = int((a-x1*1000)/100); 
	x2 = int((a-x1*1000-y1*100)/10); 
	y2 = int((a-x1*1000 - y1*100 - x2*10));

	return [x1,y1,x2,y2]; 

def xyToa(x1,y1,x2,y2):
	return x1*1000+y1*100+x2*10+y2	

def xyToa(c):
	return c[0]*1000+c[1]*100+c[2]*10+c[3]; l

def initialize(include_walls = False):


	#need transition, observation, and reward
	#need to acount for walls
	#need to know where walls are first...
	#ok so walls are l's at the edges, each is the length of three states
	if(include_walls):
		walls = [[51,61],[61,51],[52,62], [62,52],[53,63],[63,53],[56,66],[66,56],[57,67],[67,57],[58,68],[68,58]];
		walls += [[31,41],[41,31],[32,42],[42,32],[33,43],[43,33]];
		walls += [[36,46],[46,36],[37,47],[47,37],[38,48],[48,38]]; 
		walls += [[63,64],[64,63],[73,74],[74,73],[83,84],[84,83]]; 
		walls += [[65,66],[66,65],[75,76],[76,75],[85,86],[86,85]]; 
		walls += [[33,34],[34,33],[23,24],[24,23],[13,14],[14,13]]; 
		walls += [[35,36],[36,35],[25,26],[26,25],[15,16],[16,15]]; 
	else:
		walls = []; 
	
	#Transition: 10 by 10 grid, with walls
	numStates = 100*100; 
	px = [[[0 for i in range(0,numStates)] for j in range(0,5)] for k in range(0,numStates)]; 
	

	for i in range(0,numStates):
		for j in range(0,numStates):
			[cx1,cy1,rx1,ry1] = aToxy(i); 
			[cx2,cy2,rx2,ry2] = aToxy(j); 

			#make sure all movements are within a ditance of 1
			if(distance(cx1,cy1,cx2,cy2) > 1 or distance(rx1,ry1,rx2,ry2) > 1):
				continue; 

			#make sure nobody crosses any walls
			if([cx1*10+cy1,cx2*10+cy2] in walls):
				continue; 
			if([rx1*10+ry1,rx2*10+ry2] in walls):
				continue; 

			#find which actions push the robber away from the cop
				#assign them .4 each, and .2 to staying still
				#make sure to normalize

			if(cx1-cx2 == 1):
				#left
				if(distance(rx2,ry2,cx1,cy1) > distance(rx1,ry1,cx1,cy1)):
					px[i][0][j] = .4; 
				elif(distance(rx2,ry2,cx1,cy1) == distance(rx1,ry1,cx1,cy1)):
					px[i][0][j] = .2; 
			elif(cx1-cx2 == -1):
				#right
				if(distance(rx2,ry2,cx1,cy1) > distance(rx1,ry1,cx1,cy1)):
					px[i][1][j] = .4; 
				elif(distance(rx2,ry2,cx1,cy1) == distance(rx1,ry1,cx1,cy1)):
					px[i][1][j] = .2;
			elif(cy1-cy2 == 1):
				#up
				if(distance(rx2,ry2,cx1,cy1) > distance(rx1,ry1,cx1,cy1)):
					px[i][2][j] = .4; 
				elif(distance(rx2,ry2,cx1,cy1) == distance(rx1,ry1,cx1,cy1)):
					px[i][2][j] = .2;
			elif(cy1-cy2 == -1):
				#down
				if(distance(rx2,ry2,cx1,cy1) > distance(rx1,ry1,cx1,cy1)):
					px[i][3][j] = .4; 
				elif(distance(rx2,ry2,cx1,cy1) == distance(rx1,ry1,cx1,cy1)):
					px[i][3][j] = .2;
			elif(cx1 == cx2 and cy1 == cy2):
				#stay
				if(distance(rx2,ry2,cx1,cy1) > distance(rx1,ry1,cx1,cy1)):
					px[i][4][j] = .4; 
				elif(distance(rx2,ry2,cx1,cy1) == distance(rx1,ry1,cx1,cy1)):
					px[i][4][j] = .2;


		#normalize transition matrixes
		for i in range(0,numStates):
			for a in range(0,5):
				if(sum(px[i][a])==0):
					px[i][a][i] = 1; 
				else:
					px[i][a] = normalize(px[i][a]); 


		#Set observations: you can detect a robber if he is right next to you and 
		#there are no walls between you.
		#You can always tell exactly where the cop is;  
		pz = [[0 for i in range(0,numStates)] for j in range(0,numStates)]; 
		r = [-1 for i in range(0,numStates)]; 
		for i in range(0,numStates):
			[cx,cy,rx,ry] = aToxy(i); 
			if(distance(cx,cy,rx,ry) < 2 and [cx*10+cy,rx*10+ry] not in walls):
				pz[i][i] = 1000; 
				r[i] = 100; 
			else:
				for j in range(0,numStates):
					[cx2,cy2,rx2,ry2] = aToxy(j);
					if(cx == cx2 and cy == cy2):
						pz[i][j] = .1; 

		#normalize observation matrixes
		for i in range(0,numStates):
			pz[i] = normalize(pz[i]); 

		return [px,pz,r]; 
		



def generateFile(include_walls = True):

	print("Initializing Grid"); 
	if(include_walls == True):
		SFile = open ("./SARSOPTests/TagAvoidWalls100.pomdp","w"); 
	else:
		SFile = open ("./SARSOPTests/TagAvoidEmpty100.pomdp","w"); 
	num = 100*100; 
	[p,z,r] = initialize(include_walls); 



	print("Printing to File"); 


	print>>SFile,"discount: 0.95"
	print>>SFile,"values: reward"
	print>>SFile,"actions: left right up down wait"
	print>>SFile,"states: " + str(num)
	print>>SFile,"observations: " + str(num)



	#moving left, 0
	print>>SFile,""
	print>>SFile,"T:left"

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

	#stay still, 4
	print>>SFile,""
	print>>SFile,"T:wait"

	slices = [[p[y][4][x] for x in range(0,num)] for y in range(0,num)]; 

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
		tmp = "R:left:" + str(i) + ":*:* " + str(r[i]); 
		print>>SFile,tmp; 
		tmp = "R:right:" + str(i) + ":*:* " + str(r[i]);
		print>>SFile,tmp; 
		tmp = "R:up:" + str(i) + ":*:* " + str(r[i]);
		print>>SFile,tmp; 
		tmp = "R:down:" + str(i) + ":*:* " + str(r[i]);
		print>>SFile,tmp; 
		tmp = "R:wait:" + str(i) + ":*:* " + str(r[i]);
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




generateFile(True); 

#print(aToxy(121)); 
#b = aToxy(121); 
#print(xyToa(b))
				