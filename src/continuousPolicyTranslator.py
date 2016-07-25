'''
****************************************************
File: continuousPolicyTranslator.py
Written By: Luke Burks
July 2016

Input: File name, hardware flag, current pose
Output: Goal pose

Use:
Initialize with the location of the pre-generated 
continuous policy file and hardware flag.  
Call the function getNextPose() with the current 
pose argument 

Version History:
1.0.0: initial release. Only intended for use with 
	two dimensional localization problems. 


****************************************************
'''



__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



import CPerseus
from CPerseus import GM
from CPerseus import Gaussian
from CPerseus import Perseus
import numpy as np 
from scipy.stats import multivariate_normal
import copy
import math


class continuousPolicyTranslator():

	def __init__(self,fileName = "tmpalphas.npy",numBeliefs = 20, hardware = False,tag = True):
		
		if("txt" in fileName):
			self.readAlphas(fileName); 
		else:
			self.Gamma = np.load(fileName); 
		self.hardware = hardware; 
		self.nB = numBeliefs; 

		if(tag == False):
			self.setSimpleInitialBelief();
		else:
			self.setTagInitialBelief(); 

		a = Perseus(dis = False,tag = tag);
		self.pz = a.pz; 
		self.r = a.r; 
		self.delA = a.delA; 
		self.delAVar = a.delAVar;  
		self.tag = tag; 


	def continuousDot(self,a,b):

		suma = 0; 
		for k in range(0,a.size-1):
			for l in range(0,b.size):
				suma += a.Gs[k].weight*b.Gs[l].weight*multivariate_normal.pdf(b.Gs[l].mean,a.Gs[k].mean,self.covAdd(a.Gs[k].var,b.Gs[l].var)); 
		return suma; 


	def getAction(self):
		g = self.Gamma[np.argmax([self.continuousDot(j,self.B) for j in self.Gamma])]
		return g.action; 


	def extractRobMove(self,s):
		x1 = s[0]; 
		y1 = s[1]; 
		x2 = s[2]; 
		y2 = s[3];


		#Robot should always move away, with .2 chance of staying still

		if(x1 > x2):
			if(y1 < y2):
				return np.random.choice([0,2,4],p = [.4,.4,.2]); 
			elif(y2 < y1):
				return np.random.choice([0,3,4], p= [.4,.4,.2]); 
			elif(y1 == y2):
				return np.random.choice([0,2,3,4],p = [.4,.2,.2,.2]); 
		elif(x1 < x2):
			if(y1 < y2):
				return np.random.choice([1,2,4],p = [.4,.4,.2]); 
			elif(y2 < y1):
				return np.random.choice([1,3,4], p= [.4,.4,.2]); 
			elif(y1 == y2):
				return np.random.choice([1,2,3,4],p = [.4,.2,.2,.2]); 
		elif(x1 == x2):
			if(y1 < y2):
				return np.random.choice([2,0,1,4],p = [.4,.2,.2,.2]); 
			elif(y2 < y1):
				return np.random.choice([3,0,1,4], p= [.4,.2,.2,.2]); 
			elif(y1 == y2):
				return 4; 


	def distance(self,x1,y1,x2,y2):
		a = (x1-x2)*(x1-x2); 
		b = (y1-y2)*(y1-y2); 
		return math.sqrt(a+b); 

	def getNextPose(self,pose,cop):
	
		if(self.tag):
			x1 = pose[0]; 
			y1 = pose[1]; 
			x2 = pose[2]; 
			y2 = pose[3]; 
		else:
			x = pose[0]; 
			y = pose[1]; 

	
		
		action = self.getAction(); 

		#Should this be here? 
		if(action == 3): 
			action = 2; 
		elif(action == 0):
			action = 1; 
		elif(action == 1):
			action = 0; 
		elif(action == 2):
			action = 3;

		#print(action);  
		
		orient = 0; 

		if(self.hardware):
			if(self.tag):
				x1 = float(x1)*2.0; 
				y1 = float(y1)*2.0; 
				x2 = float(x2)*2.0; 
				y2 = float(y2)*2.0; 
			else:	
				x = float(x)*2.0; 
				y = float(y)*2.0; 
		
		if(self.tag):
			beta = self.extractRobMove(pose);  
			destX1 = x1 + self.delA[action][beta][0]; 
			destY1 = y1 + self.delA[action][beta][1]; 
			destX2 = x2 + self.delA[action][beta][2]; 
			destY2 = y2 + self.delA[action][beta][3]; 

			if(destX1 > 9):
				destX1 = 9; 
			elif(destX1 < 0):
				destX1 = 0; 
			if(destX2 > 9):
				destX2 = 9; 
			elif(destX2 < 0):
				destX2 = 0;
			if(destY1 > 9):
				destY1 = 9; 
			elif(destY1 < 0):
				destY1 = 0;
			if(destY2 > 9):
				destY2 = 9; 
			elif(destY2 < 0):
				destY2 = 0;

		else:
			destX = x + self.delA[action][0]; 
			destY = y + self.delA[action][1]; 

		

		if(action == 0):
			actVerb = "Left";
			orient = 180; 
		elif(action == 1):
			actVerb = "Right"; 
			orient = 0; 
		elif(action == 2):
			actVerb = "Up";
			orient = 90; 
		elif(action == 3):
			actVerb = "Down";
			orient = -90; 
		else:
			actVerb = "Wait";

		if(cop):
			if(self.tag):
				o = 0; 
				if(self.distance(x1,y1,x2,y2)):
					o = 1; 
				self.beliefUpdate(action,o); 
			else:
				self.beliefUpdate(action,int(destX*10+destY)); 



		if(self.hardware):
			if(self.tag):
				destX1 = float(destX1)/2.0; 
				destY1 = float(destY1)/2.0; 
				destX2 = float(destX2)/2.0; 
				destY2 = float(destY2)/2.0; 
			else:
				destX = float(destX)/2.0; 
				destY = float(destY)/2.0; 

		

		if(self.tag):
			if(cop):
				return [destX1,destY1,0,orient]; 
			else:
				return [destX2,destY2,0,orient]
			
		else:
			return [destX,destY,0,orient];  


	def covAdd(self,a,b):
		if(type(b) is not list):
			b = b.tolist(); 
		if(type(a) is not list):
			a = a.tolist(); 

		c = copy.deepcopy(a);

		for i in range(0,2):
			for j in range(0,2):
				c[i][j] += b[i][j]; 
		return c;  

	def setSimpleInitialBelief(self):
		self.B = GM(); 
		for i in range(0,100):
			self.B.addG(Gaussian([i/10,i%10],[[1,0],[0,1]],1)); 

	def setTagInitialBelief(self):
		self.B = GM(); 
		for i in range(0,100):
			g = Gaussian(); 
			g.mean = [0,0,i/10,i%10]; 
			g.weight = 1; 
			g.var = np.eye(4)*30; 
			self.B.addG(g); 

	def readAlphas(self,fileName):
		file = open(fileName,"r"); 
		lines = np.fromfile(fileName,sep = " "); 
		
		

		self.Gamma = []; 

		count = 0; 
		countL = len(lines); 
		while(count < countL):
			tmp = lines[count:]; 
			
			num = int(tmp[0]); 
			act = int(tmp[1]); 
			count = count + 2; 
			cur = GM(); 
			cur.action = act; 
			 

			for i in range(0,num):
				tmp = lines[count:]
				
				count = count + 7;

				mean = [int(tmp[0]),int(tmp[1])]; 
				var = [[int(tmp[2]),int(tmp[3])],[int(tmp[4]),int(tmp[5])]]; 
				weight = int(tmp[6]); 
				cur.addG(Gaussian(mean,var,weight)); 
			self.Gamma += [cur]; 


	def beliefUpdate(self,a,o):

		btmp = GM(); 

		for i in self.pz[o].Gs:
			for j in self.B.Gs:
				
				tmp = multivariate_normal.pdf(np.add(np.matrix(j.mean),np.matrix(self.delA[a][0])).tolist(),i.mean,self.covAdd(self.covAdd(i.var,j.var),self.delAVar))  
				#print(i.weight,j.weight,tmp); 
				w = i.weight*j.weight*tmp.tolist(); 

				sig = (np.add(np.matrix(i.var).I, np.matrix(self.covAdd(j.var, self.delAVar)).I)).I.tolist(); 

				#sstmp = np.matrix(i.var).I*np.transpose(i.mean) + np.matrix(self.covAdd(j.var + self.delAVar)).I*np.transpose(np.add(np.matrix(j.mean),np.matrix(delA[a])));
				sstmp1 = np.matrix(i.var).I*np.transpose(np.matrix(i.mean)); 
				sstmp2 = np.matrix(self.covAdd(j.var,self.delAVar)).I; 
				sstmp21 = np.add(np.matrix(j.mean),np.matrix(self.delA[a]));
				

				sstmp3 = sstmp1 + sstmp2*np.transpose(sstmp21);  
				smean = np.transpose(sig*sstmp3).tolist()[0]; 

				btmp.addG(Gaussian(smean,sig,w)); 
		#print(btmp.size); 
		btmp.condense(self.nB); 
		btmp.normalizeWeights();
		#btmp.display();  

		self.B = btmp; 




	def simulate(self):

		#So basically just call getNextPose until the cop catches the robber
		#Probably do a gaussian around the actual return, to simulate the hardware issuses
		x1 = 0; 
		y1 = 0; 
		x2 = 5; 
		y2 = 5; 

		while(self.distance(x1,y1,x2,y2) >= 2):
			cop = self.getNextPose([x1,y1,x2,y2],True); 
			rob = self.getNextPose([x1,y1,x2,y2],False)
			x1 = cop[0]; 
			y1 = cop[1]; 
			x2 = rob[0]; 
			y2 = rob[1]; 
			print(x1,y1,x2,y2); 





if __name__ == "__main__":
	c = continuousPolicyTranslator(fileName = "cTagAlphas2.npy",hardware = False,tag = True); 
	c.simulate(); 

	'''
	print("Check 1"); 
	print(c.getNextPose([2,2,5,5])); 
	print("Check 2"); 
	print(c.getNextPose([4,0,2,0])); 
	'''
	


	 