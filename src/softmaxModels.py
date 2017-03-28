from __future__ import division




'''
***********************************************************
File: gaussianMixtures.py
Classes: GM,Gaussian

Allows for the creation, and use of Softmax functions


***********************************************************
'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__credits__ = ["Luke Burks", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"


import numpy as np; 
import random;
from random import random; 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import warnings
import math
import copy
import time
from numpy.linalg import inv,det,svd,solve
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
from mpl_toolkits.mplot3d import Axes3D
from scipy import compress






class Softmax:


	def __init__(self,weights= None,bias = None):
		
		if(weights is None):
			self.weights = [-30,-20,-10,0]; 
		else:
			self.weights = weights;

		if(bias is None):
			self.bias = [60,50,30,0];  
		else:
			self.bias = bias; 

		self.size = len(self.weights); 

		self.alpha = 3;
		self.zeta_c = [0]*len(self.weights); 
		for i in range(0,len(self.weights)):
			self.zeta_c[i] = random()*10;  


	def Estep(self,weight,bias,prior_mean,prior_var,alpha = 0.5,zeta_c = 1,softClassNum=0):
	
		#start the VB EM step
		lamb = [0]*len(weight); 

		for i in range(0,len(weight)):
			lamb[i] = self._lambda(zeta_c[i]); 

		hj = 0;

		suma = 0; 
		for c in range(0,len(weight)):
			if(softClassNum != c):
				suma += weight[c]; 

		tmp2 = 0; 
		for c in range(0,len(weight)):
			tmp2+=lamb[c]*(alpha-bias[c])*weight[c]; 
	 
		hj = 0.5*(weight[softClassNum]-suma)+2*tmp2; 




		Kj = 0; 
		for c in range(0,len(weight)):
			Kj += lamb[c]*weight[c]*weight[c]; 
		Kj = Kj*2; 

		Kp = prior_var**-1; 
		hp = Kp*prior_mean; 

		Kl = Kp+Kj; 
		hl = hp+hj; 

		mean = (Kl**-1)*hl; 
		var = Kl**-1; 


		yc = [0]*len(weight); 
		yc2= [0]*len(weight); 

		for c in range(0,len(weight)):
			yc[c] = weight[c]*mean + bias[c]; 
			yc2[c] = weight[c]*(var + mean*mean)*weight[c] + 2*weight[c]*mean*bias[c] + bias[c]**2; 


		return [mean,var,yc,yc2]; 


	def Mstep(self,m,yc,yc2,zeta_c,alpha,steps):

		z = zeta_c; 
		a = alpha; 

		for i in range(0,steps):
			for c in range(0,len(yc)):
				z[c] = math.sqrt(yc2[c] + a**2 - 2*a*yc[c]); 

			num_sum = 0; 
			den_sum = 0; 
			for c in range(0,len(yc)):
				num_sum += self._lambda(z[c])*yc[c]; 
				den_sum += self._lambda(z[c]); 

			a = ((m-2)/4 + num_sum)/den_sum; 

		return [z,a]


	def _lambda(self, zeta_c):
		return 1 / (2 * zeta_c) * ( (1 / (1 + np.exp(-zeta_c))) - 0.5)


	def calcCHat(self,prior_mean,prior_var,mean,var,alpha,zeta_c,yc,yc2,mod):
		prior_var = np.matrix(prior_var); 
		prior_mean = np.matrix(prior_mean); 
		var_hat = np.matrix(var); 
		mu_hat = np.matrix(mean); 

		
		#KLD = 0.5*(np.log(prior_var/var) + prior_var**-1*var + (prior_mean-mean)*(prior_var**-1)*(prior_mean-mean)); 

		KLD = 0.5 * (np.log(det(prior_var) / det(var_hat)) +
							np.trace(inv(prior_var) .dot (var_hat)) +
							(prior_mean - mu_hat).T .dot (inv(prior_var)) .dot
							(prior_mean - mu_hat));


		suma = 0; 
		for c in range(0,len(zeta_c)):
			suma += 0.5 * (alpha + zeta_c[c] - yc[c]) \
	                    - self._lambda(zeta_c[c]) * (yc2[c] - 2 * alpha
	                    * yc[c] + alpha ** 2 - zeta_c[c] ** 2) \
	                    - np.log(1 + np.exp(zeta_c[c])) 
		return yc[mod] - alpha + suma - KLD + 1; 

		


	def numericalProduct(self,prior,meas,low=0,high=5,res =100,vis = True):
		
		[x,softmax] = self.plot1D(low,high,res,vis = False); 
		prod = [0 for i in range(0,len(x))]; 

		
		for i in range(0,len(x)):
			prod[i] = prior.pointEval(x[i])*softmax[meas][i]; 
		if(vis == False):
			return [x,prod]; 
		else:
			plt.plot(x,prod); 
			plt.show(); 

	def vb_update(self, measurement, prior_mean,prior_var):
        
		w = np.array(self.weights)
		b = np.array(self.bias)
		m = len(w); 
		j = measurement; 
		xis = self.zeta_c; 
		alpha = self.alpha; 
		prior_var = np.array(prior_var); 
		prior_mean = np.array(prior_mean); 
		converged = False
		EM_step = 0

		while not converged and EM_step < 10000:
			################################################################
			# STEP 1 - EXPECTATION
			################################################################
			# PART A #######################################################

			# find g_j
			sum1 = 0
			for c in range(m):
			    if c != j:
			        sum1 += b[c]
			sum2 = 0
			for c in range(m):
			    sum2 = xis[c] / 2 \
			        + self._lambda(xis[c]) * (xis[c] ** 2 - (b[c] - alpha) ** 2) \
			        - np.log(1 + np.exp(xis[c]))
			g_j = 0.5 * (b[j] - sum1) + alpha * (m / 2 - 1) + sum2

			# find h_j
			sum1 = 0
			for c in range(m):
			    if c != j:
			        sum1 += w[c]
			sum2 = 0
			for c in range(m):
			    sum2 += self._lambda(xis[c]) * (alpha - b[c]) * w[c]
			h_j = 0.5 * (w[j] - sum1) + 2 * sum2

			# find K_j
			sum1 = 0
			for c in range(m):
			    sum1 += self._lambda(xis[c]) * np.outer(w[c], (w[c]))

			K_j = 2 * sum1

			K_p = inv(prior_var)
			g_p = -0.5 * (np.log(np.linalg.det(2 * np.pi * prior_var))) \
			    + prior_mean.T .dot (K_p) .dot (prior_var)
			h_p = K_p .dot (prior_mean)

			g_l = g_p + g_j
			h_l = h_p + h_j
			K_l = K_p + K_j

			mu_hat = inv(K_l) .dot (h_l)
			var_hat = inv(K_l)

            # PART B #######################################################
			y_cs = np.zeros(m)
			y_cs_squared = np.zeros(m)
			for c in range(m):
			    y_cs[c] = w[c].T .dot (mu_hat) + b[c]
			    y_cs_squared[c] = w[c].T .dot \
			        (var_hat + np.outer(mu_hat, mu_hat.T)) .dot (w[c]) \
			        + 2 * w[c].T .dot (mu_hat) * b[c] + b[c] ** 2

            ################################################################
            # STEP 2 - MAXIMIZATION
            ################################################################
			for i in range(100):  # n_{lc}

				# PART A ######################################################
				# Find xis
				for c in range(m):
				    xis[c] = np.sqrt(y_cs_squared[c] + alpha ** 2 - 2 * alpha
				                     * y_cs[c])

				# PART B ######################################################
				# Find alpha
				num_sum = 0
				den_sum = 0
				for c in range(m):
				    num_sum += self._lambda(xis[c]) * y_cs[c]
				    den_sum += self._lambda(xis[c])
				alpha = ((m - 2) / 4 + num_sum) / den_sum

            ################################################################
            # STEP 3 - CONVERGENCE CHECK
            ################################################################
			if EM_step == 0:
			    prev_log_c_hat = -1000  # Arbitrary value

			KLD = 0.5 * (np.log(det(prior_var) / det(var_hat)) +
			             np.trace(inv(prior_var) .dot (var_hat)) +
			             (prior_mean - mu_hat).T .dot (inv(prior_var)) .dot
			             (prior_mean - mu_hat))

			sum1 = 0
			for c in range(m):
			    sum1 += 0.5 * (alpha + xis[c] - y_cs[c]) \
			        - self._lambda(xis[c]) * (y_cs_squared[c] - 2 * alpha
			        * y_cs[c] + alpha ** 2 - xis[c] ** 2) \
			        - np.log(1 + np.exp(xis[c]))

			# <>TODO: don't forget Mun - unobserved parents!
			# <>CHECK - WHY DO WE ADD +1 HERE??
			log_c_hat = y_cs[j] - alpha + sum1 - KLD + 1

			if np.abs(log_c_hat - prev_log_c_hat) < 0.001:
			    break

			prev_log_c_hat = log_c_hat
			EM_step += 1

		# Resize parameters
		if mu_hat.size == 1:
			mu_post = mu_hat[0]
		else:
			mu_post = mu_hat
		if var_hat.size == 1:
			var_post = var_hat[0][0]
		else:
			var_post = var_hat

		return mu_post, var_post, log_c_hat

	def runVB(self,prior,softClassNum):
		#For the one dimensional case only

		post = GM(); 
		weight = self.weights; 
		bias = self.bias; 
		alpha = self.alpha; 
		zeta_c = self.zeta_c; 

		for g in prior.Gs:
			prevLogCHat = -1000; 

			count = 0; 
			while(count < 100000):
				
				count = count+1; 
				[mean,var,yc,yc2] = self.Estep(weight,bias,g.mean,g.var,alpha,zeta_c,softClassNum = softClassNum);
				[zeta_c,alpha] = self.Mstep(len(weight),yc,yc2,zeta_c,alpha,steps = 100);
				logCHat = self.calcCHat(g.mean,g.var,mean,var,alpha,zeta_c,yc,yc2,mod=softClassNum); 
				if(abs(prevLogCHat - logCHat) < 0.00001):
					break; 
				else:
					prevLogCHat = logCHat; 

			post.addG(Gaussian(mean,var,g.weight*np.exp(logCHat).tolist()[0][0]))
			
		return post; 

	def runVBND(self,prior,softClassNum):
		#For the N dimensional Case
		#Note: Cannot run 1D 

		post = GM(); 

		for g in prior.Gs:
			[mu,var,logCHat] = self.vb_update(softClassNum,g.mean,g.var); 

			mu = mu.tolist(); 
			var = var.tolist(); 

			post.addG(Gaussian(mu,var,g.weight*np.exp(logCHat))); 
		return post; 

	def pointEval2D(self,softClass,point):
		top = np.exp(self.weights[softClass][0]*point[0] + self.weights[softClass][1]*point[1]); 
		bottom = 0; 
		for i in range(0,self.size):
			bottom += np.exp(self.weights[i][0]*point[0] + self.weights[i][1]*point[1]); 
		return top/bottom; 

	def plot1D(self,low=0,high = 5,res = 100,labels = None,vis = True):
		x = [(i*(high-low)/res + low) for i in range(0,res)]; 
		suma = [0]*len(x); 
		softmax = [[0 for i in range(0,len(x))] for j in range(0,len(self.weights))];  
		for i in range(0,len(x)):
			tmp = 0; 
			for j in range(0,len(self.weights)):
				tmp += math.exp(self.weights[j]*x[i] + self.bias[j]);
			for j in range(0,len(self.weights)):
				softmax[j][i] = math.exp(self.weights[j]*x[i] + self.bias[j]) /tmp;
		if(vis ==True):
			for i in range(0,len(self.weights)):
				plt.plot(x,softmax[i]); 
			plt.ylim([0,1.1])
			plt.xlim([low,high]);
			if(labels is not None):
				plt.legend(labels); 
			plt.show(); 
		else:
			return [x,softmax]; 

	def plot2D(self,low = [0,0],high = [5,5], res = 100,labels = None,vis = True):
		x, y = np.mgrid[low[0]:high[0]:(float(high[0]-low[0])/res), low[1]:high[1]:(float(high[1]-low[1])/res)]
		pos = np.dstack((x, y))  
		
		model = [[[0 for i in range(0,res)] for j in range(0,res)] for k in range(0,len(self.weights))];
		
		for m in range(0,len(self.weights)):
			for i in range(0,res):
				xx = (i*(high[0]-low[0])/res + low[0]);
				for j in range(0,res):
					yy = (j*(high[1]-low[1])/res + low[1])
					dem = 0; 
					for k in range(0,len(self.weights)):
						dem+=np.exp(self.weights[k][0]*xx + self.weights[k][1]*yy + self.bias[k]);
					model[m][i][j] = np.exp(self.weights[m][0]*xx + self.weights[m][1]*yy + self.bias[m])/dem;

		dom = [[0 for i in range(0,res)] for j in range(0,res)]; 
		for m in range(0,len(self.weights)):
			for i in range(0,res):
				for j in range(0,res):
					dom[i][j] = np.argmax([model[h][i][j] for h in range(0,len(self.weights))]); 
		if(vis):
			plt.contourf(x,y,dom,cmap = 'viridis'); 
			
			fig = plt.figure()
			ax = fig.gca(projection='3d');
			colors = ['b','r','g','y','k','b','r','g','y','k']; 
			for i in range(0,len(model)):
				ax.plot_surface(x,y,model[i],color = colors[i]); 
			
			plt.show(); 
		else:
			return x,y,dom;

	def plot4DMarginals(self, low = [0,0,0,0], high = [5,5,5,5], res = 20, dims = [2,3], labels = None,vis = True):
		x,y,z,w = np.mgrid[low[0]:high[0]:(float(high[0]-low[0])/res), low[1]:high[1]:(float(high[1]-low[1])/res), low[2]:high[2]:(float(high[2]-low[2])/res),low[3]:high[3]:(float(high[3]-low[3])/res)]
		pos = np.dstack((x,y,z,w)); 

		x3,y3 = np.mgrid[low[0]:high[0]:(float(high[0]-low[0])/res), low[1]:high[1]:(float(high[1]-low[1])/res)];

		model = np.ndarray(shape = (len(self.weights),res,res,res,res)).tolist(); 
		marg = [[[0 for i in range(0,res)] for j in range(0,res)] for k in range(0,len(self.weights))];
		for m in range(0,len(self.weights)):
			for i in range(0,res):
				xx = (i*(high[0]-low[0])/res + low[0]);
				for j in range(0,res):
					yy = (j*(high[1]-low[1])/res + low[1])
					for k in range(0,res):
						zz = (k*(high[2]-low[2])/res + low[2]); 
						for l in range(0,res):
							ww = (l*(high[3]-low[3])/res + low[3])
							dem = 0; 
							for n in range(0,len(self.weights)):
								dem+=np.exp(self.weights[n][0]*xx + self.weights[n][1]*yy + self.weights[n][2]*zz + self.weights[n][3]*ww + self.bias[n]);
							model[m][i][j][k][l] = np.exp(self.weights[m][0]*xx + self.weights[m][1]*yy + self.weights[m][2]*zz + self.weights[m][3]*ww + self.bias[m])/dem;
							marg[m][k][l] += model[m][i][j][k][l]; 

		dom = [[0 for i in range(0,res)] for j in range(0,res)]; 
		for i in range(0,res):
			for j in range(0,res):
				dom[i][j] = np.argmax([model[h][(high[0]-low[0])//2][(high[1]-low[1])//2][i][j] for h in range(0,len(self.weights))]);
				#dom[i][j] = np.argmax([marg[h][i][j] for h in range(0,len(self.weights))]);  
		if(vis):
			plt.contourf(x3,y3,dom,cmap = 'viridis'); 
			
			fig = plt.figure()
			ax = fig.gca(projection='3d');
			colors = ['b','r','g','y','k']; 
			for i in range(0,len(model)):
				ax.plot_surface(x3,y3,model[i][(high[0]-low[0])//2][(high[1]-low[1])//2],color = colors[i]);
				#ax.plot_surface(x3,y3,marg[i],color = colors[i]);  
			
			plt.show(); 
		else:
			return x3,y3,dom;


def test1DSoftmax():

	weight = [-30,-20,-10,0]; 
	bias = [60,50,30,0]; 
	softClass = 0;
	low = 0; 
	high = 5; 
	res = 100; 

	#Define Likelihood Model
	a = Softmax(weight,bias); 

	#build a prior gaussian
	prior = GM([2,4],[1,0.5],[1,0.5]); 

	#Get the posterior
	post = a.runVB(prior,softClassNum = softClass);
	

	#Plot Everything
	[x0,classes] = a.plot1D(res = res,vis = False); 
	[x1,numApprox] = a.numericalProduct(prior,softClass,low=low,high=high,res = res,vis= False); 
	
	softClassLabels = ['Far left','Left','Far Right','Right']; 
	labels = ['likelihood','prior','VB Posterior','Numerical Posterior']; 
	[x2,pri] = prior.plot(low = low, high = high,num = res,vis = False);
	[x3,pos] = post.plot(low = low, high = high,num = res,vis = False); 
	plt.plot(x0,classes[softClass]); 
	plt.plot(x2,pri);
	plt.plot(x3,pos); 
	plt.plot(x1,numApprox); 
	plt.ylim([0,1.1])
	plt.xlim([low,high])
	plt.title("Fusion of prior with: " + softClassLabels[softClass]); 
	plt.legend(labels); 
	plt.show(); 

def test2DSoftmax():
	#Specify Parameters
	#2 1D robots obs model
	#weight = [[0.6963,-0.6963],[-0.6963,0.6963],[0,0]]; 
	#bias = [-0.3541,-0.3541,0]; 
	
	#Colinear Problem
	weight = [[-1.3926,1.3926],[-0.6963,0.6963],[0,0]];
	bias = [0,.1741,0]; 
	low = [0,0]; 
	high = [5,5]; 

	#Differencing Problem
	#weight = [[0,1],[-1,1],[1,1],[0,2],[0,0]]
	#bias = [1,0,0,0,0]; 
	# low = [-5,-5]; 
	# high = [5,5]; 

	MMS = True; 
	softClass = 2; 
	detect = 0; 
	
	res = 100; 
	steep = 2; 
	for i in range(0,len(weight)):
		for j in range(0,len(weight[i])):
			weight[i][j] = weight[i][j]*steep; 
		bias[i] = bias[i]*steep; 

	#Define Likelihood Model
	a = Softmax(weight,bias);
	[x1,y1,dom] = a.plot2D(low=low,high=high,res=res,vis=False); 

	a.plot2D(low=low,high=high,res=res,vis=True); 

	#Define a prior
	prior = GM(); 
	prior.addG(Gaussian([2,4],[[1,0],[0,1]],1)); 
	prior.addG(Gaussian([4,2],[[1,0],[0,1]],1)); 
	prior.addG(Gaussian([1,3],[[1,0],[0,1]],1));
	[x2,y2,c2] = prior.plot2D(low = low,high = high,res = res, vis = False); 

	if(MMS):
		#run Variational Bayes
		if(detect == 0):
			post1 = a.runVBND(prior,0); 
			post2 = a.runVBND(prior,2); 
			post1.addGM(post2); 
		else:
			post1 = a.runVBND(prior,1); 
	else:
		post1 = a.runVBND(prior,softClass)
	post1.normalizeWeights(); 
	[x3,y3,c3] = post1.plot2D(low = low,high = high,res = res, vis = False); 
	post1.display(); 

	softClassLabels = ['Near','Left','Right','Up','Down']; 
	detectLabels = ['No Detection','Detection']
	#plot everything together
	fig,axarr = plt.subplots(3,sharex= True,sharey = True);
	axarr[0].contourf(x2,y2,c2,cmap = 'viridis'); 
	axarr[0].set_title('Prior GM'); 
	axarr[1].contourf(x1,y1,dom,cmap = 'viridis'); 
	axarr[1].set_title('Likelihood Softmax'); 
	axarr[2].contourf(x3,y3,c3,cmap = 'viridis'); 
	if(MMS):
		axarr[2].set_title('Posterior GM with observation:' + detectLabels[detect]); 
	else:
		axarr[2].set_title('Posterior GM with observation:' + softClassLabels[softClass]); 
	fig.suptitle('2D Fusion of a Gaussian Prior with a Softmax Likelihood')
	plt.show(); 


def test4DSoftmax():
	#Specify Parameters
	#2 2D robots obs model
	
	#weight = [[0.2157,-0.2271,-0.3824,0.2579],[0.3755,-0.2042,-0.2922,0.4640],[0.4088,-0.1460,-0.3254,-0.1445],[0.5417,0.2403,-0.1250,-0.2403],[0,0,0,0]]; 
	#bias = [0.4641,-0.0265,0.1397,0.1251,0]
	weight = [[-0.000000000000000,0.577350269189626,0.000000000000001,-0.577350269189625],[-0.369235473571571,0.203049310453070,0.369235473571572,-0.951651227926181],[0.530986563169930,1.028572859805121,-0.530986563169930,-0.126127678574130],[-0.625091502364266,0.279249212617170,-0.625091502364265,-0.279249212617170],[0,0,0,0]];
	bias = [-0.577350269189626,0.091325971371189,-0.407341778512092,-0.375378123870669,0]; 
	
	

	softClass = 2; 
	low = [-5,-5]; 
	high = [5,5]; 
	res = 100; 
	steep = 10; 
	for i in range(0,len(weight)):
		for j in range(0,len(weight[i])):
			weight[i][j] = weight[i][j]*steep; 
		bias[i] = bias[i]*steep; 

	#Define Likelihood Model
	a = Softmax(weight,bias);
	#[x1,y1,dom] = a.plot2D(low=low,high=high,res=res,vis=False); 
	a.plot4DMarginals(low = [-5,-5,-5,-5],high=[5,5,5,5]); 

	#a.plot2D(low=low,high=high,res=res,vis=True); 

	#Define a prior
	prior = GM(); 
	var = (np.identity(4)*4).tolist(); 
	prior.addG(Gaussian([2.5,2.5,2.5,2.5],var,1)); 
	#prior.addG(Gaussian([3,2,1,2],var,1)); 
	#prior.addG(Gaussian([2,3,2,1],var,1));
	#prior.normalizeWeights(); 
	#[x2,y2,c2] = prior.plot2D(low = low,high = high,res = res, vis = False); 

	
	post1 = a.runVBND(prior,softClass)
	#[x3,y3,c3] = post1.plot2D(low = low,high = high,res = res, vis = False); 
	

	softClassLabels = ['Left','Up','Down','Near','Right']; 
	
	#post1.normalizeWeights(); 
	priorCut = prior.slice2DFrom4D(low=low,high=high,res=res,dims = [2,3],vis=False,retGS = True); 
	postCut = post1.slice2DFrom4D(low=low,high=high,res=res,dims = [2,3],vis=False,retGS = True); 

	[x1,y1,c1] = priorCut.plot2D(low=low,high=high,res = res, vis = False); 
	[x2,y2,c2] = postCut.plot2D(low=low,high=high,res = res, vis = False); 
	
	fig,axarr = plt.subplots(2,sharex= True,sharey = True);
	axarr[0].contourf(x1,y1,c1,cmap = 'viridis'); 
	axarr[0].set_title('Prior GM'); 
	axarr[1].contourf(x2,y2,c2,cmap = 'viridis'); 
	axarr[1].set_title('Posterior GM with Observation: ' + softClassLabels[softClass]); 
	fig.suptitle('4D Fusion of a Gaussian Prior with a Softmax Likelihood, Cut to 2D')
	
	plt.show(); 

if __name__ == "__main__":

	#test1DSoftmax(); 
	test2DSoftmax(); 
	#test4DSoftmax(); 
	
	


	


