
import numpy as np; 
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/'); 
from gaussianMixtures import Gaussian
from gaussianMixtures import GM



data = np.load('../results/D2Diffs/D2Diffs_Data1.npy',encoding='latin1').tolist(); 

data2 = np.load('../results/D2DiffsSoftmax/D2DiffsSoftmax_Data1.npy',encoding='latin1').tolist(); 

data3 = np.load('../results/D2DiffsSoftmax/D2DiffsSoftmax_Data_Greedy1.npy',encoding='latin1').tolist(); 

print(data3);

rewardsReg = data['Rewards']; 
rewardsSoft = data2['Rewards']; 
rewardsGreedy = data3['Rewards']; 

averageFinalRewardReg = 0;
averageFinalRewardSoft = 0; 
averageFinalRewardGreedy = 0; 

averageAllRewardReg = [0]*len(rewardsReg[0]); 
averageAllRewardSoft = [0]*len(rewardsSoft[0]); 
averageAllRewardGreedy= [0]*len(rewardsGreedy[0]); 

for i in range(0,len(rewardsReg)):
	averageFinalRewardReg += rewardsReg[i][-1]/len(rewardsReg); 
for i in range(0,len(rewardsSoft)):
	averageFinalRewardSoft += rewardsSoft[i][-1]/len(rewardsSoft); 
for i in range(0,len(rewardsGreedy)):
	averageFinalRewardGreedy += rewardsGreedy[i][-1]/len(rewardsGreedy); 


for i in range(0,len(rewardsReg)):
	for j in range(0,len(rewardsReg[i])):
		averageAllRewardReg[j] += rewardsReg[i][j]/len(rewardsReg); 

for i in range(0,len(rewardsSoft)):
	for j in range(0,len(rewardsSoft[i])):
		averageAllRewardSoft[j] += rewardsSoft[i][j]/len(rewardsSoft); 

for i in range(0,len(rewardsGreedy)):
	for j in range(0,len(rewardsGreedy[i])):
		averageAllRewardGreedy[j] += rewardsGreedy[i][j]/len(rewardsGreedy); 




varianceReg = 0; 
SigmaReg = 0; 
allSigmaReg = [0]*len(rewardsReg[0]); 


varianceSoft = 0; 
SigmaSoft = 0; 
allSigmaSoft = [0]*len(rewardsSoft[0]); 

varianceGreedy = 0; 
SigmaGreedy = 0; 
allSigmaGreedy = [0]*len(rewardsGreedy[0]); 

suma = 0; 
for j in range(0,len(rewardsReg)):
	suma += (rewardsReg[j][-1] - averageFinalRewardReg)**2; 
varianceReg = suma/len(rewardsReg); 
SigmaReg = np.sqrt(varianceReg); 

for i in range(0,len(rewardsReg[0])):
	suma = 0; 
	for j in range(0,len(rewardsReg)):
		suma+= (rewardsReg[j][i] - averageAllRewardReg[i])**2; 
	allSigmaReg[i] = np.sqrt(suma/len(rewardsReg)); 

for i in range(0,len(rewardsSoft[0])):
	suma = 0; 
	for j in range(0,len(rewardsSoft)):
		suma+= (rewardsSoft[j][i] - averageAllRewardSoft[i])**2; 
	allSigmaSoft[i] = np.sqrt(suma/len(rewardsSoft));

for i in range(0,len(rewardsGreedy[0])):
	suma = 0; 
	for j in range(0,len(rewardsGreedy)):
		suma+= (rewardsGreedy[j][i] - averageAllRewardGreedy[i])**2; 
	allSigmaGreedy[i] = np.sqrt(suma/len(rewardsGreedy));


suma2 = 0; 
for j in range(0,len(rewardsSoft)):
	suma2 += (rewardsSoft[j][-1] - averageFinalRewardSoft)**2; 
varianceSoft = suma2/len(rewardsSoft); 
SigmaSoft = np.sqrt(varianceSoft); 

suma3 = 0; 
for j in range(0,len(rewardsGreedy)):
	suma3 += (rewardsGreedy[j][-1] - averageFinalRewardGreedy)**2; 
varianceGreedy = suma3/len(rewardsGreedy); 
SigmaGreedy = np.sqrt(varianceGreedy); 


UBReg = [0]*len(rewardsReg[0]); 
UBSoft = [0]*len(rewardsSoft[0]); 
UBGreedy = [0]*len(rewardsGreedy[0]); 

LBReg = [0]*len(rewardsReg[0]); 
LBSoft = [0]*len(rewardsSoft[0]); 
LBGreedy = [0]*len(rewardsGreedy[0]); 


for i in range(0,len(rewardsReg[0])):
	UBReg[i] = averageAllRewardReg[i] + allSigmaReg[i]; 
	LBReg[i] = averageAllRewardReg[i] - allSigmaReg[i]; 

for i in range(0,len(rewardsSoft[0])):
	UBSoft[i] = averageAllRewardSoft[i] + allSigmaSoft[i]; 
	LBSoft[i] = averageAllRewardSoft[i] - allSigmaSoft[i]; 

for i in range(0,len(rewardsGreedy[0])):
	UBGreedy[i] = averageAllRewardGreedy[i] + allSigmaGreedy[i]; 
	LBGreedy[i] = averageAllRewardGreedy[i] - allSigmaGreedy[i]; 

print(averageFinalRewardReg);
print(averageFinalRewardSoft);
print(averageFinalRewardGreedy);

print(SigmaReg);
print(SigmaSoft);
print(SigmaGreedy);


'''
Trace Plots
plt.figure(); 
for i in range(0,len(rewardsReg)):
	plt.plot(rewardsReg[i],c='r'); 
for i in range(0,len(rewardsSoft)):
	plt.plot(rewardsSoft[i],c='b'); 
for i in range(0,len(rewardsGreedy)):
	plt.plot(rewardsGreedy[i],c='g');
plt.legend(['Red: GM','Blue: Softmax','Green: Greedy']); 
plt.xlabel('Time Step'); 
plt.ylabel('Accumlated Reward'); 
plt.title('All simulation rewards'); 
plt.grid(); 
'''

x = [i for i in range(0,len(averageAllRewardReg))]; 

plt.figure(); 
plt.plot(x,averageAllRewardReg,'r'); 
plt.plot(x,averageAllRewardSoft,'b'); 
plt.plot(x,averageAllRewardGreedy,'g'); 

plt.legend(['GM-POMDP','VB-POMDP','Greedy']); 

plt.plot(x,UBReg,'r--'); 
plt.plot(x,LBReg,'r--'); 
plt.fill_between(x,LBReg,UBReg,color='r',alpha=0.25); 


plt.plot(x,UBSoft,'b--'); 
plt.plot(x,LBSoft,'b--'); 
plt.fill_between(x,LBSoft,UBSoft,color='b',alpha=0.25); 


plt.plot(x,UBGreedy,'g--'); 
plt.plot(x,LBGreedy,'g--'); 
plt.fill_between(x,LBGreedy,UBGreedy,color='g',alpha=0.25); 

plt.xlabel('Time Step'); 
plt.ylabel('Accumlated Reward'); 
plt.title('Average Accumulated Rewards over Time')

#Bar Plots
fig, ax = plt.subplots()

rects = ax.bar(np.arange(3),[averageFinalRewardReg,averageFinalRewardSoft,averageFinalRewardGreedy],yerr=[SigmaReg,SigmaSoft,SigmaGreedy])
for rect in rects:
	height = rect.get_height()
	ax.text(rect.get_x() + rect.get_width()/3., 1.05*height,
	        '%d' % int(height),
	        ha='center', va='bottom')

ax.set_xticks(np.arange(3)); 
ax.set_xticklabels(('GM-POMDP','VB-POMDP','Greedy'));
ax.set_ylabel('Average Reward'); 
ax.set_title('Average Final Rewards for Differencing Problem')

plt.show(); 






