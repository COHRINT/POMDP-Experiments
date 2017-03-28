from __future__ import division;
from sys import path
path.append('/home/luke/Documents/POMDP/src');

from gaussianMixtures import Gaussian
from gaussianMixtures import GM
from scipy.stats import multivariate_normal as mvn

import numpy as np; 
import copy; 
import matplotlib.pyplot as plt;



def continuousDot(a,b):
	suma = 0;  

	if(isinstance(a,np.ndarray)):
		a = a.tolist(); 
		a = a[0]; 

	if(isinstance(a,list)):
		a = a[0];

	a.clean(); 
	b.clean(); 

	for k in range(0,a.size):
		for l in range(0,b.size):
			suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean, np.matrix(a.Gs[k].var)+np.matrix(b.Gs[l].var)); 
	return suma; 





alpha1 = GM(); 
alpha1.addG(Gaussian(1,1,2)); 
alpha1.addG(Gaussian(2,0.5,-1)); 

alpha2 = GM(); 
alpha2.addG(Gaussian(3,0.25,0.5)); 
alpha2.addG(Gaussian(4,0.5,-1)); 

alpha3 = GM(); 
alpha3.addG(Gaussian(5,2,2)); 
alpha3.addG(Gaussian(3,0.5,-1)); 



[x,c] = alpha1.plot(low=0,high=5,vis = False); 

[x2,c2] = alpha2.plot(low=0,high=5,vis = False); 

[x3,c3] = alpha3.plot(low=0,high=5,vis = False); 


plt.plot(x,c,linewidth=5,color='r');
plt.plot(x2,c2,linewidth=5,color='g');
plt.plot(x3,c3,linewidth=5,color='y');

plt.axhline(y=0, xmin=0, xmax=5, linewidth=2, color = 'k')
plt.title('Alpha Functions'); 
plt.xlabel('Position'); 
plt.legend(['Move Right','Stay','Move Left']);
plt.show(); 




belief = GM(); 
belief.addG(Gaussian(1,1,0.5)); 
belief.addG(Gaussian(5,3,0.5)); 


[x,c] = belief.plot(low=0,high=5,vis = False); 

plt.plot(x,c,linewidth=5,color='b');
plt.axhline(y=0, xmin=0, xmax=5, linewidth=2, color = 'k')
plt.title('Belief'); 
plt.xlabel('Position'); 
plt.show(); 


fig = plt.figure(); 
 
a = GM(1,0.5,0.2); 
b = GM(4,0.5,0.1); 
c = GM(2.5,0.5,0.1); 

d = GM(); 
d.addGM(a); 
d.addGM(b); 
d.addGM(c); 

[x1,c1] = a.plot(low=0,high=5,vis=False); 
[x2,c2] = b.plot(low=0,high=5,vis=False); 
[x3,c3] = c.plot(low=0,high=5,vis=False); 
[x4,c4] = d.plot(low=0,high=5,vis=False); 

plt.plot(x1,c1,c='r',linewidth=5); 
plt.plot(x2,c2,c='g',linewidth=5); 
plt.plot(x3,c3,c='y',linewidth=5); 
plt.plot(x4,c4,c='b',linewidth=5); 
plt.title('Gaussian Mixture Components'); 
plt.xlim([0,5]); 
plt.xlabel('Position'); 
plt.legend(['Mixand 1','Mixand 2','Mixand 3','Mixture'])
plt.show(); 




fig, ax = plt.subplots()

rects = ax.bar(np.arange(3),[59,57,19],yerr=[39.6,54,55])
for rect in rects:
	height = rect.get_height()
	ax.text(rect.get_x() + rect.get_width()/3., 1.05*height,
	        '%d' % int(height),
	        ha='center', va='bottom')

ax.set_xticks(np.arange(3)); 
ax.set_xticklabels(('GM-POMDP','VB-POMDP','Greedy'));
ax.set_ylabel('Average Reward'); 
ax.set_title('Average Final Rewards for Colinear Robots')

plt.show(); 