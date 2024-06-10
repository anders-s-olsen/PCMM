clear

%load 1200 subspaces from HCP-fMRI stacked horisontally
data = h5read('C:\Users\ansol\OneDrive\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\HCP\hcp_dfc\data\processed\fMRI_SchaeferTian116_GSR_RL1.h5','/Dataset',[1,1],[2400,116]);
eigenvalues = h5read('C:\Users\ansol\OneDrive\OneDrive - Danmarks Tekniske Universitet\Dokumenter\PhD\HCP\hcp_dfc\data\processed\fMRI_SchaeferTian116_GSR_RL1.h5','/Eigenvalues',[1,1],[1200,2]);

% compute angles between all subspaces using SVD and subspace
angles = zeros(1200,1200,2); %SVD (two angles)
angles2 = zeros(1200,1200); %subspace (only the largest angle)

% Loop over time points
for i = 1:1200
    Mi = data(i*2-1:i*2,:)'; %subspace 1 (V_{t1})
    for j = 1:1200
        Mj = data(j*2-1:j*2,:)'; %subspace 2 (V_{t2})

        % Computing principal angles
        [U,S,V] = svd(Mi'*Mj);
        angles(i,j,:) = real(acos(diag(S))); 

        % Computing the subspace angle
        angles2(i,j) = subspace(Mi,Mj);
    end
end

% The subspace angle is equal to the largest of the two principal angles
figure,
subplot(2,2,1),imagesc(angles(:,:,1)),colorbar,title('Principal angle 1')
xlabel('Timepoints')
subplot(2,2,2),imagesc(angles(:,:,2)),colorbar,title('Principal angle 2')
xlabel('Timepoints')
subplot(2,2,3),imagesc(angles2),colorbar,title('Subspace angle')
xlabel('Timepoints')