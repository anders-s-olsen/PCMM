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
    dses = dir([subjects(sub).folder,'/',subjects(sub).name,'/fMRI/rfMRI_REST*']);
    for ses = 1:numel(dses)
        tic
        %         if exist(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'])&...
        %                 exist(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'])
        %             continue
        %         end
        disp(['Working on subject ',subjects(sub).name,' session ',num2str(ses)])
        data = double(squeeze(niftiread([dses(ses).folder,'/',dses(ses).name])));
        if perform_GSR
            GS = detrend(mean(data,2));
            data = data-GS.*(data'*GS)'/(GS'*GS);
        end
        %         disp(['Loaded data in ',num2str(toc),' seconds'])
        % atlas
        data_roi = nan(size(data,1),max(atlas(:)));
        eigenvectors_roi = nan(size(data,1)*2,max(atlas(:)));
        for roi = 1:max(atlas(:))
            data_roi(:,roi) = mean(data(:,atlas==roi),2);
            data_roi(:,roi) = angle(hilbert(filtfilt(bfilt,afilt,detrend(data_roi(:,roi)))));
        end
        if any(isnan(data_roi(:)))
            error('nan reached')
        end
        %         disp(['Atlas Hilbert done in ',num2str(toc),' seconds'])
        for tt = 1:size(data,1)

            cosX = cos(data_roi(tt,:));
            sinX = sin(data_roi(tt,:));

            y = data_roi(tt,:)';
            X = [cosX',sinX'];

            [U,~,~] = svds(X,2);
            eigenvectors_roi(tt*2-1:tt*2,:) = U';
        end
        disp(['Atlas eig done in ',num2str(toc),' seconds'])
        if perform_GSR
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116_GSR/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_roi)
        else
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_roi)
        end
        %         Phase_BOLD = nan(size(data));
        %         eigenvectors_all = nan(size(data));
        %
        %         for i = 1:size(data,2)
        %             Phase_BOLD(:,i) = angle(hilbert(filtfilt(bfilt,afilt,detrend(data(:,i)))));
        %         end
        % %         disp(['data Hilbert done in ',num2str(toc),' seconds'])
        %
        %         for tt = 1:size(data,1)
        %
        %             cosX = cos(Phase_BOLD(tt,:));
        %             sinX = sin(Phase_BOLD(tt,:));
        %
        %             y = Phase_BOLD(tt,:)';
        %             X = [cosX',sinX'];
        %
        %             [U,~,~] = svds(X,2);
        %             eigenvectors_all(tt*2-1:tt*2,:) = U;
        %         end
        %         disp(['Data eig done in ',num2str(toc),' seconds'])
        %
        %         parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_all)

    end
end

function parSave(fname, dopt)
save(fname, 'dopt')
end