clear all
clc; 
close all; 


B = [-1 0 1 1 0 -1 0 1 -1 0 -1 1]'; 
B = [-1 0 -1 1 0 -1 0 1 -1 0 -1 -1]'; 

M = zeros(12,15); 

%Boundry: Left|Near
rowSB = 0; 
classNum1 = 1; 
classNum2 = 0; 
for i=1:3
    M(3*rowSB + i,3*classNum2 + i) = -1; 
    M(3*rowSB + i,3*classNum1+i) = 1; 
end

%Boundry: Right|Near
rowSB = 1; 
classNum1 = 2; 
classNum2 = 0; 
for i=1:3
    M(3*rowSB + i,3*classNum2 + i) = -1; 
    M(3*rowSB + i,3*classNum1+i) = 1; 
end



%Boundry: Up|Near
rowSB = 2; 
classNum1 = 3; 
classNum2 = 0; 
for i=1:3
    M(3*rowSB + i,3*classNum2 + i) = -1; 
    M(3*rowSB + i,3*classNum1+i) = 1; 
end


%Boundry: Down|Near
rowSB = 3; 
classNum1 = 4; 
classNum2 = 0; 
for i=1:3
    M(3*rowSB + i,3*classNum2 + i) = -1; 
    M(3*rowSB + i,3*classNum1+i) = 1; 
end



A = [M,B]; 
rank(A)
rank(M)

Theta = linsolve(M,B)



