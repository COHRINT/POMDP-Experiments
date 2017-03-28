'''
Test a homegrown version of the first step of the VBIS algorithm to generate
unnormalized gaussians from gaussians and softmax models
'''
from __future__ import division
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import numpy as np
from scipy.stats import multivariate_normal as mvn
import copy
from random import random
import time
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv,det



def Estep(weight,bias,prior_mean,prior_var,alpha = 0.5,zeta_c = 1,modelNum=0):
	
	#start the VB EM step
	lamb = [0]*len(weight); 

	for i in range(0,len(weight)):
		lamb[i] = _lambda(zeta_c[i]); 

	hj = 0;

	suma = 0; 
	for c in range(0,len(weight)):
		if(modelNum != c):
			suma += weight[c]; 

	tmp2 = 0; 
	for c in range(0,len(weight)):
		tmp2+=lamb[c]*(alpha-bias[c])*weight[c]; 
 
	hj = 0.5*(weight[modelNum]-suma)+2*tmp2; 




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


def Mstep(m,yc,yc2,zeta_c,alpha,steps):

	z = zeta_c; 
	a = alpha; 

	for i in range(0,steps):
		for c in range(0,len(yc)):
			z[c] = math.sqrt(yc2[c] + a**2 - 2*a*yc[c]); 

		num_sum = 0; 
		den_sum = 0; 
		for c in range(0,len(yc)):
			num_sum += _lambda(z[c])*yc[c]; 
			den_sum += _lambda(z[c]); 

		a = ((m-2)/4 + num_sum)/den_sum; 

	return [z,a]


def _lambda(zeta):
	return (1/(2*zeta))*(1/(1+math.exp(-zeta)) - 1/2);


def calcCHat(prior_mean,prior_var,mean,var,alpha,zeta_c,yc,yc2,mod):
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
                    - _lambda(zeta_c[c]) * (yc2[c] - 2 * alpha
                    * yc[c] + alpha ** 2 - zeta_c[c] ** 2) \
                    - np.log(1 + np.exp(zeta_c[c])) 
	return yc[mod] - alpha + suma - KLD + 1; 

	


def numericalProduct(prior,likelihood,x):
	prod = [0 for i in range(0,len(likelihood))]; 

	for i in range(0,len(x)):
		prod[i] = prior.pointEval(x[i])*likelihood[i]; 
	return prod; 


def runVB(weight,bias,prior,alpha,zeta_c,modelNum):
	post = GM(); 
	
	for g in prior.Gs:
		prevLogCHat = -1000; 

		count = 0; 
		while(count < 100000):
			
			count = count+1; 
			[mean,var,yc,yc2] = Estep(weight,bias,g.mean,g.var,alpha,zeta_c,modelNum =model);
			[zeta_c,alpha] = Mstep(len(weight),yc,yc2,zeta_c,alpha,steps = 20);
			logCHat = calcCHat(g.mean,g.var,mean,var,alpha,zeta_c,yc,yc2,mod=model); 
			if(abs(prevLogCHat - logCHat) < 0.00001):
				break; 
			else:
				prevLogCHat = logCHat; 

		post.addG(Gaussian(mean,var,g.weight*np.exp(logCHat).tolist()[0][0]))
		
	return post; 


#build a softmax model
#weight = [0,15,25]; 
#bias = [0,-20,-45];
'''
weight = [0,50,100];
bias = [75,0,-125]; 
'''

weight = [-20,-10,0]; 
bias = [40,25,0]; 

'''
weight = [-30,-20,-10,0]; 
bias = [60,50,30,0]; 
'''

zeta_c = [6,2,4]; 

x = [i/10 - 5 for i in range(0,100)]; 
suma = [0]*len(x); 
softmax = [[0 for i in range(0,len(x))] for j in range(0,len(weight))];  
for i in range(0,len(x)):
	tmp = 0; 
	for j in range(0,len(weight)):
		tmp += math.exp(weight[j]*x[i] + bias[j]);
	for j in range(0,len(weight)):
		softmax[j][i] = math.exp(weight[j]*x[i] + bias[j]) /tmp;


#build a prior gaussian
prior = GM([0,-2],[1,0.5],[1,0.5]); 
model = 0;

alpha = 3;

post = runVB(weight,bias,prior,alpha,zeta_c,modelNum =model);
numApprox = numericalProduct(prior,softmax[model],x); 

modelLabels = ['left','near','right']; 
for i in range(0,len(weight)):
	plt.plot(x,softmax[i]); 
plt.ylim([0,1.1])
plt.xlim([0,5])
plt.legend(modelLabels); 
plt.title("Softmax Model for 1D Robot Localization Problem")
plt.show(); 




labels = ['likelihood','prior','VB Posterior','Numerical Posterior']; 
pri = prior.plot(low = -5, high = 5,num = len(x),vis = False);
pos = post.plot(low = -5, high = 5,num = len(x),vis = False);
plt.plot(x,softmax[model]); 
plt.plot(x,pri);
plt.plot(x,pos);  
plt.plot(x,numApprox); 
plt.ylim([0,1.1])
plt.xlim([-5,5])
plt.title("Fusion of prior with: " + modelLabels[model]); 
plt.legend(labels); 
plt.show(); 




















