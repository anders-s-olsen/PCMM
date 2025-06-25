clear
maxNumCompThreads('automatic');
rng shuffle
% subjects = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw');
subjects = readtable('/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/255unrelatedsubjectsIDs.txt');
subjects = subjects.Var1;
subfolder = '/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/raw/';

perform_GSR = true;
if perform_GSR
    add_GSR = '_GSR';
    disp('GSR will be performed')
else
    add_GSR = '';
    disp('GSR will not be performed')
end
perform_phaserandomization = false;
% task = REST, MOTOR, SOCIAL, LANGUAGE, GAMBLING, EMOTION, WM, RELATIONAL
tasks = {'EMOTION','GAMBLING','LANGUAGE','MOTOR','REST','RELATIONAL', 'SOCIAL', 'WM'};
num_add = 100;
for task = tasks
    task = task{1};
    mkdir(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR]);

    % Compute eigenvectors
    TR = 0.72;%s
    fnq=1/(2*TR);                 % Nyquist frequency
    % flp = 0.009;                    % lowpass frequency of filter (Hz)
    % fhi = 0.08;                    % highpass
    flp = 0.03;                    % lowpass frequency of filter (Hz)
    fhi = 0.07;                    % highpass
    Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
    k=2;                          % 2nd order butterworth filter
    [bfilt,afilt]=butter(k,Wn);   % construct the filter

    % atlas = squeeze(niftiread('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/external/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4.dlabel.nii'));
    atlas = squeeze(niftiread('/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/external/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii'));
    for sub = 1:numel(subjects)
        dses = dir([subfolder,num2str(subjects(sub)),'/fMRI/*fMRI_',task,'*_RL*']);
        for ses = 1:numel(dses)
            tic
            disp(['Working on subject ',num2str(subjects(sub)),' session ',num2str(ses),' of ',num2str(numel(dses)),' for task ',task])
            data = detrend(double(squeeze(niftiread([dses(ses).folder,'/',dses(ses).name]))));
            % figure,plot(data(:,100)),print('-dpng',['/dtu-compute/HCP_dFC/2023/hcp_dfc/',task,'fMRI_SchaeferTian116',add_GSR,'_',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_raw.png'])
            
            if perform_GSR
                GS = mean(data,2);
                data = data-GS.*(data'*GS)'/(GS'*GS);
            end
            % figure,plot(data(:,100)),print('-dpng',['/dtu-compute/HCP_dFC/2023/hcp_dfc/',task,'fMRI_SchaeferTian116',add_GSR,'_',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_GSR.png'])
            
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
            phases_roi = nan(size(data,1),max(atlas(:)));
            amplitude_roi = nan(size(data,1),max(atlas(:)));
            for roi = 1:max(atlas(:))
                tmp = mean(data(:,atlas==roi),2);
                % tmp2 = [2*tmp(1)-flipud(tmp(2:num_add+1));tmp;2*tmp(end)-flipud(tmp(end-num_add:end-1))];
                % tmp3 = [2*tmp(1)+flipud(tmp(2:num_add+1));tmp;2*tmp(end)+flipud(tmp(end-num_add:end-1))];
                tmp4 = [flipud(tmp(2:num_add+1));tmp;flipud(tmp(end-num_add:end-1))];
                % tmp5 = [-flipud(tmp(2:num_add+1));tmp;-flipud(tmp(end-num_add:end-1))];
                
                % tmp_filt = filtfilt(bfilt,afilt,tmp);
                % tmp2_filt = filtfilt(bfilt,afilt,tmp2);
                % tmp3_filt = filtfilt(bfilt,afilt,tmp3);
                tmp4_filt = filtfilt(bfilt,afilt,tmp4);
                % tmp5_filt = filtfilt(bfilt,afilt,tmp5);
                
                % tmp2 = tmp4_filt;
                tmp4_filt = tmp4_filt-mean(tmp4_filt);
                hil = hilbert(tmp4_filt);
                data_roi(:,roi) = tmp4_filt(num_add+1:end-num_add);
                % data_roi(:,roi) = tmp4_filt;
                hil = hil(num_add+1:end-num_add);
                % hil = hil;
                phases_roi(:,roi) = angle(hil);
                amplitude_roi(:,roi) = abs(hil);
            end
            if any(isnan(phases_roi(:)))
                error('nan reached')
            end
            % disp(num2str(num_add))
            
            eigenvectors_roi = nan(size(data,1)*2,max(atlas(:)));
            eigenvectors_real_roi = nan(size(data,1),max(atlas(:)));
            eigenvectors_imag_roi = nan(size(data,1),max(atlas(:)));
            eigenvalues_roi = nan(size(data,1)*2,1);
            for tt = 1:size(data,1)

                cosX = cos(phases_roi(tt,:));
                sinX = sin(phases_roi(tt,:));

                y = phases_roi(tt,:)';
                X = [cosX',sinX'];

                [U,S,~] = svds(X,2);
                eigenvectors_roi(tt*2-1:tt*2,:) = U';
                eigenvectors_real_roi(tt,:) = real((cosX'+1i*sinX')/norm(cosX'+1i*sinX'));
                eigenvectors_imag_roi(tt,:) = imag((cosX'+1i*sinX')/norm(cosX'+1i*sinX'));
                eigenvalues_roi(tt*2-1:tt*2) = diag(S).^2;
            end
            disp(['Atlas eig done in ',num2str(toc),' seconds'])

            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'.csv'],eigenvectors_roi)
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_evs.csv'],eigenvalues_roi)
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_real.csv'],eigenvectors_real_roi)
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_imag.csv'],eigenvectors_imag_roi)
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_amplitude.csv'],amplitude_roi)
            parSave(['/dtu-compute/HCP_dFC/2023/hcp_dfc/paper/data/processed/',task,'fMRI_SchaeferTian116',add_GSR,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_timeseries.csv'],data_roi)
                    
        end
    end
end
function parSave(fname, var)
    writetable(array2table(var),fname,'WriteVariableNames',false)
end