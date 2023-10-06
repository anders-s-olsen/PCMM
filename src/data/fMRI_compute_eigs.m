clear
maxNumCompThreads('automatic');
rng shuffle
subjects = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw');

perform_GSR = true;
perform_phaserandomization = true;

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
for sub = 1:numel(subjects)
    dses = dir([subjects(sub).folder,'/',subjects(sub).name,'/fMRI/rfMRI_REST*_RL*']);
    for ses = 1:numel(dses)
        tic
%         if perform_GSR
%             if exist(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116_GSR/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'])
%                 disp('continue')
%                 continue
%             end
%         else
%             if exist(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'])
%                 disp('continue')
%                 continue
%             end
%         end
        disp(['Working on subject ',subjects(sub).name,' session ',num2str(ses)])
        data = detrend(double(squeeze(niftiread([dses(ses).folder,'/',dses(ses).name]))));
        
        if perform_GSR
            GS = mean(data(:,1:59412),2);
            data = data-GS.*(data'*GS)'/(GS'*GS);
        end
        
        if perform_phaserandomization
            data_fft = fft(data);
            for i = 1:size(data_fft,2)
                [th,r] = cart2pol(real(data_fft(:,i)),imag(data_fft(:,i)));
                unif_phase = rand(1200,1)*2*pi;
                [re,im] = pol2cart(th+unif_phase,r);
                data_fft(:,i) = re+1i*im;
            end
            data = abs(ifft(data_fft));
        end



        data_roi = nan(size(data,1),max(atlas(:)));
        eigenvectors_roi = nan(size(data,1)*2,max(atlas(:)));
        for roi = 1:max(atlas(:))
            data_roi(:,roi) = mean(data(:,atlas==roi),2);
            data_roi(:,roi) = angle(hilbert(filtfilt(bfilt,afilt,data_roi(:,roi))));
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
            if perform_phaserandomization
                parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116_GSR_PR/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_roi)
            else
                parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116_GSR/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_roi)
            end
        else
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian116/',subjects(sub).name,'_',dses(ses).name(1:end-13),'.mat'],eigenvectors_roi)
        end
                

    end
end

function parSave(fname, dopt)
save(fname, 'dopt')
end