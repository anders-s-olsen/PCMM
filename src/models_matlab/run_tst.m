clear
rng(0);
%% synth data 1 (simple, Watson)
data1 = [1,1,1]+randn(1000,3)*0.02;
data1(1:500,:)=-data1(1:500,:);
data2 = [1,-1,-1]+randn(1000,3)*0.02;
data2(1:500,:)=-data2(1:500,:);
data = [data1;data2];
data = data./vecnorm(data,2,2);

writetable(array2table(data),'synth_data_1.csv','WriteVariableNames',false)

%% synth data 2 (from oscillating functions
clear, rng(0);
n = 1000;
timeaxis = 0:0.1:99.9; %100 seconds, TR=0.1
data1 = cos(timeaxis+0.1*randn(1,length(timeaxis)))+0.1*randn(1,length(timeaxis));
data2 = cos(timeaxis+0.1*randn(1,length(timeaxis)))+0.1*randn(1,length(timeaxis));
data3(1:500) = cos(timeaxis(1:500)+0.1*randn(1,length(timeaxis)/2))+0.1*randn(1,length(timeaxis)/2);
data3(501:1000) = cos(timeaxis(501:end)+pi+0.1*randn(1,length(timeaxis)/2))+0.1*randn(1,length(timeaxis)/2);

data = [data1',data2',data3'];

figure, hold on
for p = 1:3
plot(data(:,p)+2.1*p)
end

datahil = angle(hilbert(data));
for i = 1:n
    cosmap = cos(datahil(i,:)-datahil(i,:)');
    [V(:,(i-1)*2+1:(i-1)*2+2),l] = eigs(cosmap,2);
    if l(1)<l(2)
        error('eigenvectors not sorted')
    end
end

figure, hold on
for p = 1:3
plot(datahil(:,p)+(2*pi+0.1)*p)
end

figure, hold on
for p = 1:3
plot(V(2:2:2*n,p)+1.1*p)
end

V = V';
writetable(array2table(V),'synth_data_2.csv','WriteVariableNames',false)


%%
pointsspherefig(V(1:2:2*n,:),[ones(500,1),2*ones(500,1)])
for i = 1:100
    rng("shuffle")
    results = WMM_EM_BigMem2(V(1:2:2*n,:),2,1000,1,'uniform');
    kaps(:,i) = results.kappa;
end
% showcase matlab results
contourspherefig(results.mu,results.kappa')