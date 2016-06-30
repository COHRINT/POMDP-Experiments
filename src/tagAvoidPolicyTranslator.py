'''
************************************************************
File: tagAvoidPolicyTranslator.py
Written By: Luke Burks
May 2016


Input: Filename, hardware flag, current poses
Output: Goal poses

The tagAvoidPolicyTranslator class is designed to 
process a policy for a simplified version of the
tag avoid problem.



Use: 
Initialize with the file name containing the policy,
represented by alpha vectors with the desired 
cardinal action as the last element, and a flag 
to indicate hardware simulations. 
The desired pose is generated by the getNextPose() 
function with the current pose of each 
robot as arguments. 
The getNextPose() function can make use of the 'secondary'
argument which forces the cop to take its second best action
according to it's policy. 
In simulation, the belief is initialized as a symetric gaussian
around the actual location. A grid Bayes filter is used to 
update the belief.


Example:
sim = tagAvoidPolicyTranslator("tagAvoidEmpty100.txt",True);
tmp = getNextPose(copPose,robberPose); 
'Send Cop tmp[0]'
'Send Robber tmp[1]'

Version history:
0.8: Inital Testing, arbitary number
0.9: Added actual belief handling and proper observations
<<<<<<< HEAD
0.91: Added belief update within getCopPose()
=======

>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
*************************************************************
'''


__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "0.91"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



import numpy as np
from discretePolicyTranslator import discretePolicyTranslator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import random
from scipy.stats import multivariate_normal
import copy
import math
from math import sqrt

#TODO: Run a RandomEmpty policy
#TODO: The belief doesn't seem to be taking into account the robbers movements
#TODO: Do both animations at once
#TODO: Figure out a way to deal with wait motions that doesn't involve going through walls
#NOTE: You changed the observation model to an adjacent squares model
#NOTE: You changed the getNextCopPose() function to switch 2 and 3

class tagAvoidPolicyTranslator(discretePolicyTranslator):


	def __init__(self,fileName = None,hardware = False):
		if(fileName == None):
			self.alphas = self.readAlphas("fakealphas1.txt");
		else:
			self.alphas = self.readAlphas(fileName); 
		if(isinstance(self.alphas[0],float)):
			self.numStates = len(self.alphas)-1;
		else:
			self.numStates = len(self.alphas[0])-1;
		self.goalX= 0; 
		self.goalY = 0;  
		self.hardware = hardware; 
		self.belief = [1.0 for i in range(0,100)];
<<<<<<< HEAD
		if('Walls' in fileName or 'walls' in fileName):
			self.walls = True; 
		else:
			self.walls = False; 
=======

>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		





	#Specialized for this program, not general for the sake of computation time
	def gridBayesFilter(self,cx,cy,b,u,z,walls = []):
		


		#For each i, there are only 3 non-zero entries in px[i][u]
		#So just multiply each one with the corresponding b entry, using the xyToa


		bNew = [0 for i in range(0,len(b))];
		
		for i in range(0,len(b)):
			#get the positions
			rx = i/10; 
			ry = i%10; 


			#find and catalog all non zero new positions
			if(u == 0 and cx > 0):
				destC = [cx-1,cy]; 
			elif(u == 1 and cx < 9):
				destC = [cx+1,cy]; 
			elif(u == 3 and cy < 9):
				destC = [cx,cy+1]; 
			elif(u == 2 and cy > 0):
				destC = [cx,cy-1];  
			else:
				destC = [cx,cy]; 
			
			dests = [[destC[0],destC[1],rx,ry]]; 
			weights = [.2]; 
			if(cx-rx >= 0 and rx < 9 and [rx*10 + ry,(rx+1)*10+ry] not in walls):
				dests += [[destC[0],destC[1],rx+1,ry]];
				weights += [.4];  
			elif(cx-rx<=0 and rx > 0 and [rx*10 + ry,(rx-1)*10+ry] not in walls):
				dests += [[destC[0],destC[1],rx-1,ry]]
				weights += [.4]; 
			if(cy-ry >= 0 and ry < 9 and [rx*10 + ry,(rx)*10+ry+1] not in walls):
				dests += [[destC[0],destC[1],rx,ry+1]]; 
				weights += [.4]; 
			elif(cy-ry <= 0 and ry > 0 and [rx*10 + ry,(rx)*10+ry-1] not in walls):
				dests += [[destC[0],destC[1],rx,ry-1]];
				weights += [.4]; 

			#print(dests); 
			#convert them with xyToa
			states = []; 

			for j in dests:
				states += [j[2]*10 + j[3]]; 

			#also find their probabilities by checking if the robber moved then its .4 else .2
			#normalize those probabilities
			weights = self.normalize(weights); 

			total = 0; 
			#multiply each one by the corresponding xyToa entry in b

			#print(len(states)); 
			#print(len(weights)); 
			#print(len(b)); 

			for j in range(0,len(states)):
				#print("State: " + str(states[j]) + " J: " + str(j) + " Weight: " + str(weights[j]) + " Belief: " + str(b[states[j]])); 
				total += weights[j]*b[states[j]]; 
			if(cx == rx and cy == ry and z == 1):
				bNew[i] = total; 
			elif((cx != rx or cy != ry) and z == 0):
				bNew[i] = total;
			else:
				bNew[i] = 0; 

			if(self.distance(cx,cy,rx,ry) == 0 and z == 2):
				bNew[i] = total; 
			elif(self.distance(cx,cy,rx,ry) < 2 and z == 1):
				bNew[i] = total; 
			elif(self.distance(cx,cy,rx,ry) > 2 and z == 0):
				bNew[i] = total; 

		bNew = self.normalize(bNew); 

		return bNew;


	def getProbBelief(self,mean,var):
		if(var < .01):
			var = .01; 
		cov = [[var,0],[0,var]]; 
		b = [0 for i in range(0,100)];
		for i in range(0,100):
			b[i] = multivariate_normal.pdf([i%10,i/10],mean,cov); 
		b = self.normalize(b);
		return b; 


	def normalize(self,a):

		Suma = sum(a); 
		for i in range(0,len(a)):
			a[i] = float(a[i])/Suma; 
		return a; 

	def aToxy(self,a):
		x1 = int(a/1000);
		y1 = int((a-x1*1000)/100); 
		x2 = int((a-x1*1000-y1*100)/10); 
		y2 = int((a-x1*1000 - y1*100 - x2*10));

		return [x1,y1,x2,y2]; 

	def xyToa(self,x1,y1,x2,y2):
		return x1*1000+y1*100+x2*10+y2	

	def xyToa(self,c):
		return c[0]*1000+c[1]*100+c[2]*10+c[3]; 


	def distance(self,x1,y1,x2,y2):
		a = (x1-x2)*(x1-x2); 
		b = (y1-y2)*(y1-y2); 
		return sqrt(a+b); 

	def fakeBelief(self,x1,y1,x2,y2):
		arr = [0]*self.numStates; 
		a = self.xyToa([x1,y1,x2,y2]); 
		arr[a] = 1; 
		return arr;  


	def printMap(self,cx,cy,rx,ry):
		map1 = ""; 
		for i in range(9,-1,-1):
			for j in range(0,10):
				if(j == cx and i == cy):
					map1 = map1+'O'; 
				elif(j==rx and i == ry):
					map1 = map1+'X'; 
				else:
					map1 = map1+"-"; 
			map1 = map1+"\n"; 
		print(map1); 

	def getMOMDPAction(self,alphas,belief,pose):
		bestValue = -10000000; 
		bestAction = 0; 

		for i in range(0,len(alphas)):
			if(alphas[i][len(alphas[i])-1] != pose):
				continue; 
			total = 0; 
			for j in range(0,len(belief)):
				total = total + belief[j]*alphas[i][j]; 
			if(total>bestValue):
				bestValue = total; 
				bestAction = alphas[i][len(belief)];  
		
		return bestAction; 


	def getNextCopPose(self,copPose,robberPose,secondary=False,sayAction = False, bel= [],walls = []):


		if(self.hardware):
			cx = int(round(copPose[0]*2)); 
			cy = int(round(copPose[1]*2)); 
			rx = int(round(robberPose[0]*2)); 
			ry = int(round(robberPose[1]*2)); 
		else:
			cx = int(copPose[0]); 
			cy = int(copPose[1]); 
			rx = int(robberPose[0]); 
			ry = int(robberPose[1]); 

		a = [cx,cy,rx,ry]; 

		if(bel == []):
<<<<<<< HEAD
			belief = self.belief; 
=======
			belief = self.fakeBelief(cx,cy,rx,ry); 
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		else:
			belief = bel; 
 		
		action = self.getMOMDPAction(self.alphas,belief,cx*10+cy); 

		#if(secondary == False):
			
		#else:
			#action = self.getSecondaryAction(self.alphas,belief);



		
		orient = 0.0; 

		#TODO: The up and down are switched. Maybe change that back? 

		if(action == 0):
			destX = cx-1; 
			destY = cy; 
			actVerb = "Left"; 
			orient = 180.0; 
		elif(action == 1):
			destX = cx+1; 
			destY = cy; 
			actVerb = "Right"; 
			orient = 0.0; 
		elif(action == 3):
			destX = cx; 
			destY = cy+1; 
			actVerb = "Up";
			orient = 90.0; 
		elif(action == 2):
			destX = cx; 
			destY = cy-1; 
			actVerb = "Down";
			orient = -90.0; 
		else:
			destX = cx; 
			destY = cy;
			actVerb = "Wait";

		

		if(self.hardware):
			z = 0; 
<<<<<<< HEAD
			if(self.distance(destX,destY,rx,ry) ==0):
				z = 2; 
			elif(self.distance(destX,destY,rx,ry) < 2):
=======
			if(distance(destX,destY,rx,ry) ==0):
				z = 2; 
			elif(distance(destX,destY,rx,ry) < 2):
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
				z = 1; 
			self.belief = self.gridBayesFilter(destX,destY,self.belief,action,z,walls); 

			return [float(destX)/2,float(destY)/2,0.0,orient]; 
		else:
			return [destX,destY,0,orient,action]

	def getNextPose(self,copPose,robberPose,secondary = False, walls = []):
		cop = self.getNextCopPose(copPose,robberPose,secondary, walls = walls); 
		rob = self.getNextRobberPose(copPose,robberPose); 
		return [cop,rob]; 

	def getNextRobberPose(self,copPose,robberPose):

		if(self.hardware):
			cx = int(round(copPose[0]*2)); 
			cy = int(round(copPose[1]*2)); 
			rx = int(round(robberPose[0]*2)); 
			ry = int(round(robberPose[1]*2)); 
		else:
			cx = int(copPose[0]); 
			cy = int(copPose[1]); 
			rx = int(robberPose[0]); 
			ry = int(robberPose[1]); 

		weights = []; 
		dests = []; 
		holds = []; 
		

		walls = []
<<<<<<< HEAD
		if(self.walls):
			walls += [[51,61],[61,51],[52,62], [62,52],[53,63],[63,53],[56,66],[66,56],[57,67],[67,57],[58,68],[68,58]];
			walls += [[31,41],[41,31],[32,42],[42,32],[33,43],[43,33]];
			walls += [[36,46],[46,36],[37,47],[47,37],[38,48],[48,38]]; 
			walls += [[63,64],[64,63],[73,74],[74,73],[83,84],[84,83]]; 
			walls += [[65,66],[66,65],[75,76],[76,75],[85,86],[86,85]]; 
			walls += [[33,34],[34,33],[23,24],[24,23],[13,14],[14,13]]; 
			walls += [[35,36],[36,35],[25,26],[26,25],[15,16],[16,15]]; 
			walls += [[30,40],[40,30],[39,49],[49,39],[50,60],[60,50],[59,69],[69,59]]; #closes off rooms
			
=======
		'''
		walls += [[51,61],[61,51],[52,62], [62,52],[53,63],[63,53],[56,66],[66,56],[57,67],[67,57],[58,68],[68,58]];
		walls += [[31,41],[41,31],[32,42],[42,32],[33,43],[43,33]];
		walls += [[36,46],[46,36],[37,47],[47,37],[38,48],[48,38]]; 
		walls += [[63,64],[64,63],[73,74],[74,73],[83,84],[84,83]]; 
		walls += [[65,66],[66,65],[75,76],[76,75],[85,86],[86,85]]; 
		walls += [[33,34],[34,33],[23,24],[24,23],[13,14],[14,13]]; 
		walls += [[35,36],[36,35],[25,26],[26,25],[15,16],[16,15]]; 
		walls += [[30,40],[40,30],[39,49],[49,39],[50,60],[60,50],[59,69],[69,59]]; #closes off rooms
		'''
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f

		if(cx-rx <= 0 and rx < 9 and [rx*10 + ry,(rx+1)*10+ry] not in walls):
			dests += [[rx+1,ry,1]];
			weights += [.4];  
			holds += [1]; 
		elif(cx-rx>=0 and rx > 0 and [rx*10 + ry,(rx-1)*10+ry] not in walls):
			dests += [[rx-1,ry,0]]
			weights += [.4]; 
			holds +=[0]; 
		if(cy-ry <= 0 and ry < 9 and [rx*10 + ry,(rx)*10+ry+1] not in walls):
			dests += [[rx,ry+1,2]]; 
			weights += [.4]; 
			holds+=[2]; 
		elif(cy-ry >= 0 and ry > 0 and [rx*10 + ry,(rx)*10+ry -1] not in walls):
			dests += [[rx,ry-1,3]];
			weights += [.4]; 
			holds +=[3]

		dests += [[rx,ry,4]]; 
		weights += [.2]; 
		holds += [4]; 


		weights = self.normalize(weights); 

		tmp = np.random.choice(holds,p=weights);
		dest = dests[holds.index(tmp)];  
		orient = 0.0; 
		if(dest[2] == 0):
			orient = 180.0; 
		elif(dest[2] == 1):
			orient == 0.0; 
		elif(dest[2] == 2):
			orient = 90.0; 
		elif(dest[2] == 3):
			orient = -90.0; 
		else: 
			orient = 0.0; 

		if(self.hardware):
			return [float(dest[0])/2,float(dest[1])/2,0.0,orient]; 
		else:
			return [dest[0],dest[1],0,orient]; 


	def simulate(self):
		cx = 1; 
<<<<<<< HEAD
		cy = 9; 
		rx = 8; 
		ry = 4; 
=======
		cy = 1; 
		rx = 8; 
		ry = 1; 
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f

		self.copsx = []; 
		self.copsy = []; 
		self.robsx = []; 
		self.robsy = []; 
		self.copsx += [cx]; 
		self.copsy += [cy]; 
		self.robsx += [rx]; 
		self.robsy += [ry]; 
		self.bet = []; 
		#startTime = time.clock(); 
		walls = []; 
		
<<<<<<< HEAD
		if(self.walls):
			walls += [[51,61],[61,51],[52,62], [62,52],[53,63],[63,53],[56,66],[66,56],[57,67],[67,57],[58,68],[68,58]];
			walls += [[31,41],[41,31],[32,42],[42,32],[33,43],[43,33]];
			walls += [[36,46],[46,36],[37,47],[47,37],[38,48],[48,38]]; 
			walls += [[63,64],[64,63],[73,74],[74,73],[83,84],[84,83]]; 
			walls += [[65,66],[66,65],[75,76],[76,75],[85,86],[86,85]]; 
			walls += [[33,34],[34,33],[23,24],[24,23],[13,14],[14,13]]; 
			walls += [[35,36],[36,35],[25,26],[26,25],[15,16],[16,15]]; 
			walls += [[30,40],[40,30],[39,49],[49,39],[50,60],[60,50],[59,69],[69,59]]; #closes off rooms
			
=======
		'''
		walls += [[51,61],[61,51],[52,62], [62,52],[53,63],[63,53],[56,66],[66,56],[57,67],[67,57],[58,68],[68,58]];
		walls += [[31,41],[41,31],[32,42],[42,32],[33,43],[43,33]];
		walls += [[36,46],[46,36],[37,47],[47,37],[38,48],[48,38]]; 
		walls += [[63,64],[64,63],[73,74],[74,73],[83,84],[84,83]]; 
		walls += [[65,66],[66,65],[75,76],[76,75],[85,86],[86,85]]; 
		walls += [[33,34],[34,33],[23,24],[24,23],[13,14],[14,13]]; 
		walls += [[35,36],[36,35],[25,26],[26,25],[15,16],[16,15]]; 
		walls += [[30,40],[40,30],[39,49],[49,39],[50,60],[60,50],[59,69],[69,59]]; #closes off rooms
		'''
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f

		cov = [[12,0],[0,12]]; 
		belief = [1.0 for i in range(0,100)];

<<<<<<< HEAD
		'''
=======
		
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		for i in range(0,100):
			rx2 = i/10; 
			ry2 = i%10; 
			belief[i] = multivariate_normal.pdf([rx2,ry2],[rx,ry],cov); 
<<<<<<< HEAD
		'''
=======
		
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f

		belief = self.normalize(belief);  
		tmpb = []
		
		action = 0; 

		flag = False; 
		count = 0; 
		while(self.distance(cx,cy,rx,ry) >= 2 or [cx*10+cy,rx*10+ry] in walls):
			a = self.getNextRobberPose([cx,cy],[rx,ry]);
			b = self.getNextCopPose([cx,cy],[rx,ry],sayAction = True,bel = belief, walls = walls);
			#if([cx*10+cy,b[0]*10+b[1]] in walls):
				#b = self.getNextCopPose([cx,cy],[rx,ry],secondary = True,sayAction = True,bel = belief);
			action = b[4]; 

			count+=1; 
			
			
			'''
			if(a[0] < 0):
				a[0] = 0; 
			if(a[0] > 9):
				a[0] = 9; 
			if(a[1] < 0):
				a[1] = 0; 
			if(a[1] > 9):
				a[1] = 9; 
			'''

			if(b[0] < 0):
				b[0] = 0; 
			if(b[0] > 9):
				b[0] = 9; 
			if(b[1] < 0):
				b[1] = 0; 
			if(b[1] > 9):
				b[1] = 9; 	

			rx = a[0]; 
			ry = a[1]; 
			cx = b[0]; 
			cy = b[1]; 

			'''
			if(cx < 0 or cy < 0 or ry < 0 or rx < 0 or cx > 9 or cy > 9 or rx > 9 or ry > 9):
				print("Error: Robot out of bounds"); 
				flag = True; 
				break; 
			'''

			if(self.distance(cx,cy,rx,ry) == 0):
				z = 2; 
			elif(self.distance(cx,cy,rx,ry) < 2):
				z = 1;  
			elif(self.distance(cx,cy,rx,ry) > 2):
				z = 0; 


			tmpb = []; 
			belief = self.gridBayesFilter(cx,cy,belief,action,z,walls); 
			for i in range(0,100):
				tmpb += [belief[i]]; 
			tmpb = self.normalize(tmpb); 

			self.bet += [tmpb]; 
			self.copsx += [cx]; 
			self.copsy += [cy]; 
			self.robsx += [rx]; 
			self.robsy += [ry]; 

			
			

			print("Cop position: "); 
			print(cx,cy);  
			print("Robber position: "); 
			print(rx,ry); 
			print("Action: " + str(action)); 
			self.printMap(cx,cy,rx,ry); 
			print(""); 
			#plt.scatter([cx,rx],[cy,ry]); 
			#plt.axis([-.5,9.5,-.5,9.5])
			#plt.show(); 
			#print("Time Elapsed: " + str(time.clock()-startTime)); 
			#if(count > 40):
				#break; 



		print("Congratulations!! The cop caught the robber in: " + str(count) + " moves.")



def convertVectorToGrid(b):
	a = [[0 for i in range(0,10)] for j in range(0,10)]; 
	for i in range(0,100):
		a[i/10][i%10] = b[i]; 
	return a; 

def convertGridToVector(b):
	a = []; 
	for i in b:
		a += copy.deepcopy(i); 
	return a; 



def scatterWalls():
	wallsRight = [83,73,63,85,75,65,33,23,13,35,25,15]; 
	wallsUp = [51,52,53,56,57,58,31,32,33,36,37,38]; 
	ys = [1,2,3,3.5,3.5,3.5,5.5,5.5,5.5,6,7,8,1,2,3,3.5,3.5,3.5,5.5,5.5,5.5,6,7,8,3.5,5.5,3.5,5.5];  #closes off rooms
	xs = [5.5,5.5,5.5,6,7,8,8,7,6,5.5,5.5,5.5,3.5,3.5,3.5,3,2,1,1,2,3,3.5,3.5,3.5,9,9,0,0]; #closes off rooms

	'''
	xs = []; 
	ys = []; 
	for i in range(0,len(wallsRight)):
		xs += [float(int(wallsRight[i]/10))]; 
		ys += [float(wallsRight[i]%10)]; 
	for i in range(0,len(wallsUp)):
		xs += [float(int(wallsUp[i]/10))]; 
		ys += [float(wallsUp[i]%10)]; 
	'''
	ax.scatter(xs,ys,c = 'black', s = 300, marker = 's')
	


def update(data,line): 
<<<<<<< HEAD
	global cbar
=======
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
	mat.set_data(data);
	global copsx; 
	global copsy;
	global robsx;
	global robsy; 
	global bet; 
	global plotBelief
	if(len(copsx) > 0):
		#a = ax.scatter(tmp%10,tmp/10,c='black',s = 50,marker='x'); 
<<<<<<< HEAD
		#if(not plotBelief):
		plt.cla();  

		ax.scatter(robsy[0],robsx[0], c= 'red', s = 100);
			
		ax.scatter(copsy[0],copsx[0], c= 'blue', s = 100); 
=======
		if(not plotBelief):
			plt.cla();  

			ax.scatter(robsx[0],robsy[0], c= 'red', s = 100);
			
			ax.scatter(copsx[0],copsy[0], c= 'blue', s = 100); 
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		#if(plotBelief):
			#ax.scatter(copsx[0],copsy[0], c= 'white', s = 100); 

		scatterWalls(); 
<<<<<<< HEAD
		plt.imshow(convertVectorToGrid(bet[0]), interpolation = "none",cmap = 'viridis'); 
=======

>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		plt.axis([-.5, 9.5,-.5, 9.5])
		copsx = copsx[1:]; 
		copsy = copsy[1:];
		robsx = robsx[1:]; 
		robsy = robsy[1:]; 
<<<<<<< HEAD
		bet = bet[1:]; 
=======
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		
	return mat; 

def data_gen():
	global bet; 
	global size;
	global plotBelief;  
	while len(bet) > 0:
<<<<<<< HEAD
		'''
=======
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		if(plotBelief):
			yield convertVectorToGrid(bet[0]); 
		else:
			yield convertVectorToGrid([.01 for i in range(0,100)]); 
		bet = bet[1:]; 
<<<<<<< HEAD
		'''
		yield convertVectorToGrid([.01 for i in range(0,100)]);
=======
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
		


def convertVectorToString(slices):
	slices = str(slices); 
	slices = slices.replace(']',''); 
	slices = slices.replace(',','');
	slices = slices.replace('[',''); 
	return slices;

if __name__ == "__main__":


	


	
	#file = "../policies/tagAvoidEmpty100.txt"; 
	#file = "../policies/tagAvoidWalls100.txt";
	#file = "../policies/tagRandomWalls100.txt"; 
	#file = "../policies/tagAvoidWallsProper100.txt"; 
	#file = "../policies/tagAvoidWallsReduced100.txt"; 
	#file = "../policies/tagAvoidWallsStarting100.txt"; 
	#file = "../policies/tagAvoidWallsStarting2.txt"; 
	#file = "../policies/tagAvoidWallsStarting.txt"; 
	#file = "../policies/tagAvoidWallsStarting3.txt";
	#file = "../policies/tagAvoidEmptyStarting1.txt";
	#file = "../policies/tagAvoidWallsStarting1.txt";
	#file = "../policies/tagAvoidWallsStarting2.txt";
<<<<<<< HEAD
	#file = "../policies/tagAvoidEmpty2.txt"
	#file = "../policies/tagAvoidWalls4.txt"
	file = "../policies/tagAvoidWalls5.txt"
=======
	file = "../policies/tagAvoidEmpty2.txt"
	#file = "../policies/tagAvoidWalls4.txt"
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f

	t = tagAvoidPolicyTranslator(file); 
	print("Policy Loaded"); 

	#cop = [1,1]; 
	#rob = [2,2]; 
	#print(t.getNextCopPose(cop,rob)); 
	#print(t.getNextRobberPose(cop,rob));
	
	'''
	for i in range(0,30):
		cop = [random.random()*11-1,random.random()*11-1]; 
		rob = [random.random()*11-1,random.random()*11-1]; 
		print(t.getNextCopPose(cop,rob)); 
		print(t.getNextRobberPose(cop,rob)); 
	'''
	
	'''
	b = [1 for i in range(0,10000)]; 
	b = t.normalize(b); 
	print(t.gridBayesFilter(b,2,1))
	'''




	
	t.simulate()

	print("Simulation Complete"); 

	copsx = t.copsx; 
	copsy = t.copsy; 
	robsx = t.robsx; 
	robsy = t.robsy; 
	bet = t.bet; 
	plotBelief = True

<<<<<<< HEAD

	copsx = copsx[1:]; 
	copsy = copsy[1:];
	robsx = robsx[1:]; 
	robsy = robsy[1:]; 
=======
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
	'''
	for i in range(0,len(bet)):
		for j in range(0,len(bet[0])):
			bet[i][j] = math.log(bet[i][j]); 
	'''


	fig = plt.figure()
	ax = plt.axes(xlim=(-.5, 9.5), ylim=(-.5, 9.5))
	line, = ax.plot([], [], lw=2)
	colormin = min(convertGridToVector(bet)); 
	colormax = max(convertGridToVector(bet[0:len(bet)])); 
<<<<<<< HEAD
=======
	mat = ax.matshow(convertVectorToGrid(bet[0]),vmin = colormin, vmax = colormax)
>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f

	mat = ax.matshow(convertVectorToGrid(bet[0]),vmin = colormin, vmax = colormax, cmap = 'viridis')

	plt.colorbar(mat); 
	ani = animation.FuncAnimation(fig, update,frames=data_gen, interval=500, save_count=100, blit=False, fargs = [line]); 
	#plt.show(); 		
	ani.save('tagAvoidBelief.gif', writer='imagemagick', fps=2)
	
	'''
	copsx = t.copsx; 
	copsy = t.copsy; 
	robsx = t.robsx; 
	robsy = t.robsy; 
	bet = t.bet; 
	plotBelief = False
	fig = plt.figure()
	ax = plt.axes(xlim=(-.5, 9.5), ylim=(-.5, 9.5))
	line, = ax.plot([], [], lw=2)
	colormin = min(convertGridToVector(bet)); 
	colormax = max(convertGridToVector(bet)); 
	mat = ax.matshow(convertVectorToGrid(bet[0]),vmin = colormin, vmax = .5)

<<<<<<< HEAD
	cbar = plt.colorbar(mat); 
	ani = animation.FuncAnimation(fig, update,frames=data_gen, interval=500, save_count=100, blit=False, fargs = [line]); 
	#plt.show(); 		
	ani.save('tagAvoidBelief.gif', writer='imagemagick', fps=2)
	
	'''
	copsx = t.copsx; 
	copsy = t.copsy; 
	robsx = t.robsx; 
	robsy = t.robsy; 
	bet = t.bet; 
	plotBelief = False
	fig = plt.figure()
	ax = plt.axes(xlim=(-.5, 9.5), ylim=(-.5, 9.5))
	line, = ax.plot([], [], lw=2)
	colormin = min(convertGridToVector(bet)); 
	colormax = max(convertGridToVector(bet)); 
	mat = ax.matshow(convertVectorToGrid(bet[0]),vmin = colormin, vmax = .5)

	#plt.colorbar(mat); 
	ani = animation.FuncAnimation(fig, update,frames=data_gen, interval=100, save_count=100, blit=False, fargs = [line]); 

=======
	#plt.colorbar(mat); 
	ani = animation.FuncAnimation(fig, update,frames=data_gen, interval=100, save_count=100, blit=False, fargs = [line]); 

>>>>>>> 441b412aca9fd7dcddeb996e66012e4da0ee2c1f
	ani.save('tagAvoidPlain.gif', writer='imagemagick', fps=2)
	'''




	'''
	PolFile = open("../policies/tagAvoidWallsStarting2.txt","w"); 
	fileName = "../policies/tagAvoidWallsStarting100.txt"; 
	alphas = t.readAlphas(fileName); 

	for al in alphas:
		tmp = []; 
		for i in range(0,100):
			tmp += [al[i]]; 
		tmp = convertVectorToString(tmp) + " " + str(al[len(al)-1]); 
		print>> PolFile,tmp; 
	'''
	
