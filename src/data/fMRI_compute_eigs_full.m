clear
maxNumCompThreads('automatic');
rng shuffle
subjects = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw');

perform_GSR = true;

% Compute eigenvectors
TR = 0.72;%s
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.009;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

% atlas = squeeze(niftiread('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/external/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4.dlabel.nii'));
atlas = squeeze(niftiread('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/external/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii'));
for sub = randperm(numel(subjects))
    dses = dir([subjects(sub).folder,'/',subjects(sub).name,'/fMRI/rfMRI_REST*_RL*']);
    for ses = 1:numel(dses)
        tic
        if perform_GSR
            if exist(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full_GSR/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'])
                disp('continue')
                continue
            end
        else
            if exist(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'])
                disp('continue')
                continue
            end
        end
        disp(['Working on subject ',subjects(sub).name,' session ',num2str(ses)])
        data = detrend(double(squeeze(niftiread([dses(ses).folder,'/',dses(ses).name]))));

        if perform_GSR
            GS = mean(data(:,1:59412),2);
            data = data-GS.*(data'*GS)'/(GS'*GS);
        end
        Phase_BOLD = nan(size(data));
        eigenvectors_all = nan(size(data));

        for i = 1:size(data,2)
            Phase_BOLD(:,i) = angle(hilbert(filtfilt(bfilt,afilt,data(:,i))));
        end
        %         disp(['data Hilbert done in ',num2str(toc),' seconds'])

        for tt = 1:size(data,1)

            cosX = cos(Phase_BOLD(tt,:));
            sinX = sin(Phase_BOLD(tt,:));

            y = Phase_BOLD(tt,:)';
            X = [cosX',sinX'];

            [U,~,~] = svds(X,2);
            eigenvectors_all(tt*2-1:tt*2,:) = U';
        end
        disp(['Data eig done in ',num2str(toc),' seconds'])
        if perform_GSR
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full_GSR/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_all)
        else
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_all)
        end

    end
end

function parSave(fname, dopt)
save(fname, 'dopt')
end