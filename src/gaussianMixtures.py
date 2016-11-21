from __future__ import division

import numpy as np; 
import random;
from random import random; 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import warnings


class Gaussian:
	def __init__(self,u = None,sig = None,w=1):
		warnings.filterwarnings("ignore")
		if(u == None):
			self.mean = [0,0]; 
		else:
			self.mean = u; 
		if(sig == None):
			self.sig = [[1,0],[0,1]];
		else:
			self.var = sig; 
		self.weight = w; 

	def display(self):
		print("Mean: ");
		print(self.mean); 
		print("Variance: "); 
		print(self.var); 
		print("Weight"); 
		print(self.weight); 

class GM:
	def __init__(self,u=None,s=None,w=None):
		self.Gs = []; 
		if(w == None):
			self.size = 0; 
		elif(isinstance(w,float) or isinstance(w,int)):
			self.size = 0; 
			self.addG(Gaussian(u,s,w)); 
		elif(len(w) > 1):
			for i in range(0,len(w)):
				self.Gs += [Gaussian(u[i],s[i],w[i])];
		self.size = len(self.Gs);  
		self.action = None; 


	def getMeans(self):
		ans = []; 
		for g in self.Gs:
			ans += [[g.mean]]; 
		return ans; 

	def getVars(self):
		ans = []; 
		for g in self.Gs:
			ans += [[g.var]]; 
		return ans; 

	def getWeights(self):
		ans = []; 
		for g in self.Gs:
			ans += [g.weight]; 
		return ans; 



	def clean(self):

		for g in self.Gs:
			if(not isinstance(g.mean,list)):
				g.mean = g.mean.tolist(); 

			if(not isinstance(g.var,list)):
				g.var = g.var.tolist(); 

			while(len(g.mean) != len(g.var)):
				g.mean = g.mean[0]; 

			for i in range(0,len(g.var)):
				g.var[i][i] = abs(g.var[i][i]); 



	def findMAP2D(self):
		[a,b,res] = self.plot2D(vis=False); 
		MAP= [0,0]; 
		meanVal = [-10000]; 
		for i in range(0,len(res)):
			for j in range(0,len(res[i])):
				if(res[i][j] > meanVal):
					meanVal = res[i][j]; 
					MAP = [i/20,j/20]; 
		return MAP; 


	def plot(self,low = -20,high = 20,num = 1000,vis = True):
		a = np.linspace(low,high,num= num); 
		b = [0.0]*num; 
		for g in self.Gs:
			b += mvn.pdf(a,g.mean,g.var)*g.weight; 
		if(vis):
			plt.plot(a,b);   
			plt.show(); 
		else:
			return b; 

	def plot2D(self,low = [0,0], high = [5,5],vis = True,res = 100,xlabel = 'Cop Belief',ylabel = 'Robber Belief',title = 'Belief'):
		
		c = [[0 for i in range(0,res)] for j in range(0,res)]; 		

		x, y = np.mgrid[low[0]:high[0]:(float(high[0])/res), low[1]:high[1]:(float(high[1])/res)]

		pos = np.dstack((x, y)) 
		
		self.clean(); 

		for g in self.Gs:
			
			try:
				c += mvn.pdf(pos,g.mean,g.var)*g.weight; 
			except:
				g.display();
				raise; 


		if(vis):
			fig,ax = plt.subplots(); 
			ax.contourf(x,y,c,cmap = 'viridis'); 
			#fig.colorbar(); 
			ax.set_xlabel(xlabel); 
			ax.set_ylabel(ylabel);
			ax.set_title(title);
			plt.show(); 
		else:
			return x,y,c; 

	

	def normalizeWeights(self):
		suma = 0; 
		for g in self.Gs:
			suma += g.weight; 
		for g in self.Gs:
			g.weight = g.weight/suma; 
		self.size = len(self.Gs); 

	def addGM(self,b):
		for i in range(0,len(b.Gs)):
			self.addG(b.Gs[i]);
		self.size = len(self.Gs); 

	def addG(self,b):
		self.Gs += [b];
		self.size+=1; 
		self.size = len(self.Gs); 

	def display(self):
		 
		print("Means"); 
		print([self.Gs[i].mean for i in range(0,self.size)]); 
		print("Variances"); 
		print([self.Gs[i].var for i in range(0,self.size)]); 
		print("Weights"); 
		print([self.Gs[i].weight for i in range(0,self.size)]); 
		if(self.action is not None):
			print("Action"); 
			print(self.action); 

	def comp(self,b):
		if(self.size != b.size):
			return False; 

		for g in range(0,self.size):
			if(self.Gs[g].mean != b.Gs[g].mean):
				return False; 
			if(self.Gs[g].weight != b.Gs[g].weight):
				return False; 

			if(isinstance(self.Gs[g].var,(int,float))):
				if(self.Gs[g].var != b.Gs[g].var):
					return False; 
			else:
				for i in self.Gs[g].var:
					if(i not in b.Gs[g].var):
						return False; 
					'''
			if(self.Gs[g].var != b.Gs[g].var):
				return False; 
				'''

		return True; 

	def pointEval(self,x):
		suma = 0; 
		self.clean(); 
		for g in self.Gs:
			suma += g.weight*mvn.pdf(x,g.mean,g.var); 
		return suma; 


	def distance2D(self,a,b):
		ans = (a[0] - b[0])**2 + (a[1]-b[1])**2; 
		return ans; 


	def kmeansCondensation(self,k = 10,lowInit = [0,0], highInit = [5,5]):
		#only pursues a single clustering step, does not attempt to converge

		means = [0]*k; 
		for i in range(0,k):
			#means[i] = [(i/k)*(highInit[0]-lowInit[0]) - lowInit[0],  (i/k)*(highInit[1]-lowInit[1]) - lowInit[1]]; 
			means[i] = [random()*(highInit[0]-lowInit[0]) - lowInit[0],  random()*(highInit[1]-lowInit[1]) - lowInit[1]]; 

		
		clusters = [GM() for i in range(0,k)]; 


		for g in self.Gs:
			#put the gaussian in the cluster which minimizes the distance between the distribution mean and the cluster mean
			clusters[np.argmin([self.distance2D(g.mean,means[j]) for j in range(0,k)])].addG(g);  
			#print(np.argmin([distance2D(g.mean,means[j]) for j in range(0,k)]))
		
			

		#reorient the means
		for i in range(0,k):
			suma = [0,0]; 
			for g in clusters[i].Gs:
				suma[0] += g.mean[0]; 
				suma[1] += g.mean[1]; 
			if(clusters[i].size != 0):
				suma[0] = suma[0]/clusters[i].size; 
				suma[1] = suma[1]/clusters[i].size; 
			means[i] = suma; 


		#condense each cluster
		for c in clusters:
			c.condense(1); 

		#add each cluster back together
		ans = GM(); 
		for c in clusters:
			ans.addGM(c);  

		ans.action = self.action; 

		#return big cluster
		return ans; 



	def printClean(self,slices):
		slices = str(slices); 
		slices = slices.replace(']',''); 
		slices = slices.replace(',','');
		slices = slices.replace('[',''); 
		return slices;

	def printToFile(self,file):
		#first line is N, number of gaussians
		#next N lines are, mean, variance, weight
		file.write(str(self.size) + " " + str(self.action) + "\n"); 
		for g in self.Gs:
			m = self.printClean(g.mean); 
			var = self.printClean(g.var); 
			w = self.printClean(g.weight); 
			file.write(m + " " + var + " " + w + "\n"); 



	def scalerMultiply(self,s):
		for g in self.Gs:
			g.weight = s*g.weight; 

	def condense(self, max_num_mixands=None):
       
		
		if max_num_mixands is None:
			max_num_mixands = self.max_num_mixands

		

		#Check if any mixands are small enough to not matter
		#specifically if they're weighted really really low
		dels = []; 
		for g in self.Gs:
			if(g.weight < 0.000001):
				dels.append(g);

		for rem in dels:
			if(rem in self.Gs):
				self.Gs.remove(rem);
				self.size = self.size-1; 

		#Check if merging is useful
		if self.size <= max_num_mixands:
		    return


		# Create lower-triangle of dissimilarity matrix B
		#<>TODO: this is O(n ** 2) and very slow. Speed it up! parallelize?
		B = np.zeros((self.size, self.size))
	 
		for i in range(self.size):
		    mix_i = (self.Gs[i].weight, self.Gs[i].mean, self.Gs[i].var) 
		    for j in range(i):
		        if i == j:
		            continue
		        mix_j = (self.Gs[j].weight, self.Gs[j].mean, self.Gs[j].var) 
		        B[i,j] = self.mixand_dissimilarity(mix_i, mix_j)
		       	

		# Keep merging until we get the right number of mixands
		deleted_mixands = []
		toRemove = []; 
		while self.size > max_num_mixands:
		    # Find most similar mixands
		   
			try:
				min_B = B[B>0].min()
			except:
				self.display(); 
				raise; 



			ind = np.where(B==min_B)
			i, j = ind[0][0], ind[1][0]

			# Get merged mixand
			mix_i = (self.Gs[i].weight, self.Gs[i].mean, self.Gs[i].var) 
			mix_j = (self.Gs[j].weight, self.Gs[j].mean, self.Gs[j].var) 
			w_ij, mu_ij, P_ij = self.merge_mixands(mix_i, mix_j)

			# Replace mixand i with merged mixand
			ij = i
			self.Gs[ij].weight = w_ij
			self.Gs[ij].mean = mu_ij.tolist(); 
			self.Gs[ij].var = P_ij.tolist(); 



			# Fill mixand i's B values with new mixand's B values
			mix_ij = (w_ij, mu_ij, P_ij)
			deleted_mixands.append(j)
			toRemove.append(self.Gs[j]);

			#print(B.shape[0]); 

			for k in range(0,B.shape[0]):
			    if k == ij or k in deleted_mixands:
			        continue

			    # Only fill lower triangle
			   # print(self.size,k)
			    mix_k = (self.Gs[k].weight, self.Gs[k].mean, self.Gs[k].var) 
			    if k < i:
			        B[ij,k] = self.mixand_dissimilarity(mix_k, mix_ij)
			    else:
			        B[k,ij] = self.mixand_dissimilarity(mix_k, mix_ij)

			# Remove mixand j from B
			B[j,:] = np.inf
			B[:,j] = np.inf
			self.size -= 1


		# Delete removed mixands from parameter arrays
		for rem in toRemove:
			if(rem in self.Gs):
				self.Gs.remove(rem); 
		

		

	def mixand_dissimilarity(self,mix_i, mix_j):
		"""Calculate KL descriminiation-based dissimilarity between mixands.
		"""
		# Get covariance of moment-preserving merge
		w_i, mu_i, P_i = mix_i
		w_j, mu_j, P_j = mix_j
		_, _, P_ij = self.merge_mixands(mix_i, mix_j)

		#TODO: This is different
		if(w_i < 0 and w_j< 0):
			w_i = abs(w_i); 
			w_j = abs(w_j); 


		if(P_ij.ndim == 1 or len(P_ij.tolist()[0]) == 1):
				if(not isinstance(P_ij,(int,list,float))):
					P_ij = P_ij.tolist()[0];
				while(isinstance(P_ij,list)):
					P_ij = P_ij[0];

				if(not isinstance(P_i,(int,list,float))):
					P_i = P_i.tolist()[0];  
				while(isinstance(P_i,list)):
					P_i = P_i[0];
				if(not isinstance(P_j,(int,list,float))):
					P_j = P_j.tolist()[0];
				while(isinstance(P_j,list)):
					P_j = P_j[0];
					


				logdet_P_ij = P_ij; 
				logdet_P_i = P_i; 
				logdet_P_j = P_j; 


		else:
		    # Use slogdet to prevent over/underflow
		    _, logdet_P_ij = np.linalg.slogdet(P_ij)
		    _, logdet_P_i = np.linalg.slogdet(P_i)
		    _, logdet_P_j = np.linalg.slogdet(P_j)
		    
		    # <>TODO: check to see if anything's happening upstream
		    if np.isinf(logdet_P_ij):
		        logdet_P_ij = 0
		    if np.isinf(logdet_P_i):
		        logdet_P_i = 0
		    if np.isinf(logdet_P_j):
		        logdet_P_j = 0

		#print(logdet_P_ij,logdet_P_j,logdet_P_i)

		b = 0.5 * ((w_i + w_j) * logdet_P_ij - w_i * logdet_P_i - w_j * logdet_P_j)

		return b

	def merge_mixands(self,mix_i, mix_j):
	    """Use moment-preserving merge (0th, 1st, 2nd moments) to combine mixands.
	    """
	    # Unpack mixands
	    w_i, mu_i, P_i = mix_i
	    w_j, mu_j, P_j = mix_j

	    mu_i = np.array(mu_i); 
	    mu_j = np.array(mu_j); 

	    P_j = np.matrix(P_j); 
	    P_i = np.matrix(P_i); 

	    # Merge weights
	    w_ij = w_i + w_j
	    w_i_ij = w_i / (w_i + w_j)
	    w_j_ij = w_j / (w_i + w_j)

	    # Merge means

	    mu_ij = w_i_ij * mu_i + w_j_ij * mu_j

	    P_j = np.matrix(P_j); 
	    P_i = np.matrix(P_i); 


	    # Merge covariances
	    P_ij = w_i_ij * P_i + w_j_ij * P_j + \
	        w_i_ij * w_j_ij * np.outer(self.subMu(mu_i,mu_j), self.subMu(mu_i,mu_j))



	    return w_ij, mu_ij, P_ij

	def subMu(self,a,b):

		if(isinstance(a,np.ndarray)):
			return a-b;  
		if(isinstance(a,(float,int))):
			return a-b; 
		else:
			c = [0]*len(a); 
			for i in range(0,len(a)):
				c[i] = a[i]-b[i]; 
			return c; 






if __name__ == "__main__":

	g1 = GM(); 
	var = [[1,.8],[.8,1]]; 


	for i in range(-1,7):
		g1.addG(Gaussian([i,i],var,1)); 

	g2 = GM(); 
	for i in range(1,13):
		for j in range(-1,7):
			g2.addG(Gaussian([j,j+i],var,1)); 
			g2.addG(Gaussian([j,j-i],var,1)); 

	#gSum = g1.addGM(g2); 



	c = [[0 for i in range(0,500)] for j in range(0,500)]; 	
	d = [[0 for i in range(0,500)] for j in range(0,500)]; 	
	e = [[0 for i in range(0,500)] for j in range(0,500)]; 	

	x, y = np.mgrid[0:5:0.01, 0:5:0.01]



	pos = np.dstack((x, y)) 
	
	for g in g1.Gs:
		c += mvn.pdf(pos,g.mean,g.var)*g.weight; 
	
	
	for g in g2.Gs: 
		d += mvn.pdf(pos,g.mean,g.var)*g.weight;
	


	
	fig1 = plt.figure(); 
	plt.contourf(x,y,c,cmap = 'viridis'); 
	plt.colorbar(); 

	fig2 = plt.figure(); 
	plt.contourf(x,y,d,cmap = 'viridis'); 
	plt.colorbar(); 

	fig3 = plt.figure(); 
	plt.contourf(x,y,c+d,cmap = 'viridis'); 
	plt.colorbar(); 
	plt.show(); 


