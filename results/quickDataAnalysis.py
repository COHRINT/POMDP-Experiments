
import numpy as np; 
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/'); 
from gaussianMixtures import Gaussian
from gaussianMixtures import GM



data = np.load('./D2Diffs/D2Diffs_Data99.npy').tolist(); 

data2 = np.load('./D2DiffsSoftmax/D2DiffsSoftmax_Data99.npy').tolist(); 


print(sum(data['Rewards']));
print(data2['Rewards']); 