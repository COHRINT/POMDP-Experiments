from __future__ import division


def orient(a,b,c):
	#asks if three points are clockwise oriented
	tau = (b[1]-a[1])*(c[0]-b[0]) - (c[1]-b[1])*(b[0]-a[0]); 

	if(tau < 0):
		return False;
	else:
		return True; 


def checkIntercept(a,b):
	o1 = orient(a[0],a[1],b[0]); 
	o2 = orient(a[0],a[1],b[1]); 
	o3 = orient(b[0],b[1],a[0]); 
	o4 = orient(b[0],b[1],a[1]); 

	if(o1 != o2 and o3 != o4):
		return True; 
	else:
		return False; 


a = [[1,1],[2,2]]; 

b = [[2,1],[1,2]]; 

print(checkIntercept(a,b)); 



