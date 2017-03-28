

%For initial of 1000
finals = [1,5,10,100,200,300,500,800,999]; 
runs = [94,101,96,94,94,84,72,48,33]; 
kmeans = [93,27,12,1.2,0.6,0.5,0.4,0.4,0.5]; 

figure(); 
plot(finals,runs,'LineWidth',4); 
hold on; 
plot(finals,kmeans,'LineWidth',4); 

legend('Runnals Only','Kmeans+Runnals'); 
title('Condensation Times for a GM with 1000 Gaussians'); 
ylabel('Runtime (s)'); 
xlabel('Final Number of Gaussians, after condensation'); 


