from __future__ import division

import math
import random
import matplotlib.pyplot as plt
import copy; 
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn




a = np.linspace(-10,10,num = 1000); 


mean1 = np.matrix([0,0]).T; 
mean2 = np.matrix([1,1]).T; 

var1 = np.matrix([[1,0],[0,1]]); 
var2 = np.matrix([[2,1],[1,2]]); 

const = np.matrix([[10,0],[0,10]]); 




var3 = (var2 + const*var1*const.T).I * var1*var2;  
mean3 = (var2 + const*var1*const.T).I * (var2*mean1 + var1*const*mean2); 

#var3 = (var1.I + var2.I).I; 
#mean3 = var3*(var1.I*mean1 + var2.I*mean2); 


#Univariate case
#b = norm.pdf(a,mean1,np.sqrt(var1))norm.pdf(const*a,mean2,np.sqrt(var2));
#c = norm.pdf(mean1,mean2,np.sqrt(const*var1*const.T+var2))*norm.pdf(a,mean3,np.sqrt(var3));  
'''
f, axarr = plt.subplots(3, sharex=False)

axarr[0].plot(a,b[0]); 
axarr[1].plot(a,c[0]);
axarr[2].plot(a,d[0]); 
'''
#d = b-c; 


#Multvariate


 

mean1 = [0,0]; 
mean2 = [1,1]; 




mean3 = mean3.T.tolist()[0]; 

x, y = np.mgrid[-1:1:0.01, -1:1:0.01]




pos = np.dstack((x, y))


 

rv1 = mvn(mean1, var1);
rv2 = mvn(mean2,var2); 

rv3 = mvn(mean2,const*var1*const.T+var2); 
rv4 = mvn(mean3,var3); 

z1 = rv1.pdf(pos)*rv2.pdf(10*pos); 
z2 = rv3.pdf(mean1)*rv4.pdf(pos); 
print(rv3.pdf(mean1)); 

fig2 = plt.figure();

ax2 = fig2.add_subplot(3,1,1);
ax2.contourf(x, y, z1); 

ax3 = fig2.add_subplot(3,1,2); 
ax3.contourf(x,y,z2)

ax4 = fig2.add_subplot(3,1,3); 
ax4.contourf(x,y,z1-z2)





#b = mvn.pdf(a,mean1,var1)*mvn.pdf(const*a,mean2,var2);
#c = mvn.pdf(mean1,mean2,const*var1*const.T+var2)*mvn.pdf(a,mean3,var3);  




  

plt.show(); 