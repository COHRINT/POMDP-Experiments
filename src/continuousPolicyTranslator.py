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



class continuousPolicyTranslator():

	def __init__(self,fileName= "tmpalphas.npy",numBeliefs = 20, hardware = False):
		
		if("txt" in fileName):
			self.readAlphas(fileName); 
		else:
			self.Gamma = np.load(fileName); 
		self.hardware = hardware; 
		self.setInitialBelief();
		self.nB = numBeliefs; 
		self.B.condense(self.nB); 
		a = Perseus(dis = False);
		self.pz = a.pz; 
		self.r = a.r; 
		self.delA = a.delA; 
		self.delAVar = [[1,0],[0,1]]; 

	def continuousDot(self,a,b):

		suma = 0; 
		for k in range(0,a.size-1):
			for l in range(0,b.size):
				suma += a.Gs[k].weight*b.Gs[l].weight*multivariate_normal.pdf(b.Gs[l].mean,a.Gs[k].mean,self.covAdd(a.Gs[k].var,b.Gs[l].var)); 
		return suma; 


	def getAction(self):
		g = self.Gamma[np.argmax([self.continuousDot(j,self.B) for j in self.Gamma])]
		return g.action; 

	def getNextPose(self,pose):
	
		x = pose[0]; 
		y = pose[1]; 

		if(self.hardware):
			x = x*2; 
			y = y*2; 
		
		action = self.getAction(); 

		orient = 0; 

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

		self.goalX = destX; 
		self.goalY = destY; 


		if(self.hardware):
			destX = destX/2; 
			destY = destY/2; 

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

	def setInitialBelief(self):
		self.B = GM(); 
		for i in range(0,100):
			self.B.addG(Gaussian([i/10,i%10],[[1,0],[0,1]],1)); 

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
				tmp = multivariate_normal.pdf(np.add(np.matrix(j.mean),np.matrix(self.delA[a])).tolist(),i.mean,self.covAdd(self.covAdd(i.var,j.var),self.delAVar));  
				w = i.weight*j.weight*tmp; 

				sig = (np.add(np.matrix(i.var).I, np.matrix(self.covAdd(j.var, self.delAVar)).I)).I.tolist(); 

				#sstmp = np.matrix(i.var).I*np.transpose(i.mean) + np.matrix(self.covAdd(j.var + self.delAVar)).I*np.transpose(np.add(np.matrix(j.mean),np.matrix(delA[a])));
				sstmp1 = np.matrix(i.var).I*np.transpose(np.matrix(i.mean)); 
				sstmp2 = np.matrix(self.covAdd(j.var,self.delAVar)).I; 
				sstmp21 = np.add(np.matrix(j.mean),np.matrix(self.delA[a]));
				

				sstmp3 = sstmp1 + sstmp2*np.transpose(sstmp21);  
				smean = np.transpose(sig*sstmp3).tolist()[0]; 

				btmp.addG(Gaussian(smean,sig,w)); 
		btmp.condense(self.nB); 
		btmp.normalizeWeights(); 

		self.B = btmp; 




	#def simulate(self):





if __name__ == "__main__":
	c = continuousPolicyTranslator(fileName = "localizationAlphas2.npy"); 

	print("Check 1"); 
	print(c.getNextPose([2,2])); 
	print("Check 2"); 
	print(c.getNextPose([8,0])); 
	 