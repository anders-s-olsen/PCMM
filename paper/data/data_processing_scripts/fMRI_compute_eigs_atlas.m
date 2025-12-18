clear
maxNumCompThreads('automatic');
rng shuffle

%%%%%% OPTIONS %%%%%%
perform_GSR = true;
perform_9p = false;
perform_phaserandomization = false;
atlases = {'SchaeferTian116'}; %'SchaeferTian116','SchaeferTian232', 'Schaefer400'
dataset = '_clean_rclean_tclean'; %'Atlas','Atlas_MSMAll', 'rclean_tclean' (Atlas_MSMAll no longer includes both rest and task)

add_dataset = ''; % deprecated

if perform_9p && perform_GSR
    error('Cannot perform both 9p regression and GSR. Choose one or none.')
end

if perform_GSR
    add_denoising = '_GSR';
    disp('GSR will be performed')
elseif perform_9p
    add_denoising = '_9p';
    disp('9p regression will be performed')
else
    add_denoising = '';
    disp('GSR will not be performed')
end

subjects = readtable('paper/data/255unrelatedsubjectsIDs.txt');
subjects = subjects.Var1;
subfolder = 'paper/data/raw/';

% check that current working directory has a folder called 'paper'
cwd = pwd;
if ~exist([cwd,'/paper'],'dir')
    error('Current working directory does not have a folder called "paper". Please change to the correct directory.')
end

% task = REST, MOTOR, SOCIAL, LANGUAGE, GAMBLING, EMOTION, WM, RELATIONAL
tasks = {'RELATIONAL', 'SOCIAL', 'WM'}; % ,'EMOTION','GAMBLING','LANGUAGE','MOTOR','REST','RELATIONAL', 'SOCIAL', 'WM'
num_pad = 100;

for atlas_idx = atlases
    if atlas_idx{1} == "SchaeferTian116"
        atlas = squeeze(niftiread('paper/data/external/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii'));
    elseif atlas_idx{1} == "SchaeferTian232"
        atlas = squeeze(niftiread('paper/data/external/Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2.dlabel.nii'));
    elseif atlas_idx{1} == "Schaefer400"
        atlas = squeeze(niftiread('paper/data/external/Schaefer2018_400Parcels_7Networks_order.dlabel.nii'));
    end

    for task = tasks
        task = task{1};
        mkdir(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising]);

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

        
        for sub = 1:numel(subjects)
            dses = dir([subfolder,num2str(subjects(sub)),'/fMRI/*fMRI_',task,'*_RL_*',dataset,'.dtseries.nii']);
            for ses = 1:numel(dses)
                tic
                disp(['Working on subject ',num2str(subjects(sub)),' session ',num2str(ses),' of ',num2str(numel(dses)),' for task ',task, ' using atlas ',atlas_idx{1}])
                data = detrend(double(squeeze(niftiread([dses(ses).folder,'/',dses(ses).name]))));
                
                if perform_GSR
                    GS = mean(data,2);
                    data = data-GS.*(data'*GS)'/(GS'*GS);
                elseif perform_9p
                    motion_params = table2array(readtable([subfolder,num2str(subjects(sub)),'/regressors/',dses(ses).name(1:end-13),'_RL_Movement_Regressors_dt.txt']));
                    WM = table2array(readtable([subfolder,num2str(subjects(sub)),'/regressors/',dses(ses).name(1:end-13),'_RL_WM.txt']));
                    CSF = table2array(readtable([subfolder,num2str(subjects(sub)),'/regressors/',dses(ses).name(1:end-13),'_RL_CSF.txt']));
                    nineP = [motion_params(:,1:6), WM, CSF];
                    nineP = (nineP - mean(nineP,1))./std(nineP,[],1); % z-score
                    data = data - nineP*(data'*nineP)'/(nineP'*nineP);
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
                phases_roi = nan(size(data,1),max(atlas(:)));
                amplitude_roi = nan(size(data,1),max(atlas(:)));
                for roi = 1:max(atlas(:))
                    roi_data = mean(data(:,atlas==roi),2);
                    roi_data_padded = [flipud(roi_data(2:num_pad+1));roi_data;flipud(roi_data(end-num_pad:end-1))];
                    roi_data_padded_filt = filtfilt(bfilt,afilt,roi_data_padded);
                    roi_data_padded_filt_demeaned = roi_data_padded_filt-mean(roi_data_padded_filt);
                    hil = hilbert(roi_data_padded_filt_demeaned);
                    data_roi(:,roi) = roi_data_padded_filt_demeaned(num_pad+1:end-num_pad);
                    hil = hil(num_pad+1:end-num_pad);
                    phases_roi(:,roi) = angle(hil);
                    amplitude_roi(:,roi) = abs(hil);
                end
                if any(isnan(phases_roi(:)))
                    error('nan reached')
                end
                
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

                % disp(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'.csv'])
                parSave(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'.csv'],eigenvectors_roi)
                parSave(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_evs.csv'],eigenvalues_roi)
                parSave(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_real.csv'],eigenvectors_real_roi)
                parSave(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_imag.csv'],eigenvectors_imag_roi)
                parSave(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_amplitude.csv'],amplitude_roi)
                parSave(['paper/data/processed/',add_dataset,task,'fMRI_',atlas_idx{1},add_denoising,'/',num2str(subjects(sub)),'_',dses(ses).name(1:end-13),'_timeseries.csv'],data_roi)
                        
            end % session
        end % subject
    end % task
end % atlas
function parSave(fname, var)
    writetable(array2table(var),fname,'WriteVariableNames',false)
end