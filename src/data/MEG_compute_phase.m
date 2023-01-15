clear,close all

% configCluster;
% c = parcluster(dccClusterProfile());
% c.AdditionalProperties;
% c.AdditionalProperties.MemUsage = '20GB';
% c.AdditionalProperties.ProcsPerNode = 16;
% c.AdditionalProperties.WallTime = '24:00';
% c.AdditionalProperties.EmailAddress = 'ansol@dtu.dk';
% c.AdditionalProperties.QueueName = 'computebigbigmem';
% c.AdditionalProperties.AdditionalSubmitArgs = '';
% c.AdditionalProperties.EnableDebug = 0;
% c.AdditionalProperties.GpuModel = '';
% c.AdditionalProperties.GpusPerNode = 0;
% c.AdditionalProperties.JobPlacement = 'spread';
% c.AdditionalProperties.UseSmpd = 0;
% c.saveProfile
% 
% profname = dccClusterProfile();
% clust = parcluster(profname);
% % clust = parcluster('local');
% p=parpool(clust,16);

maxNumCompThreads('automatic');
subs = dir('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/raw/');
subs(1:2) = [];
for i = 1:numel(subs)
    numses(i) = numel(dir([subs(i).folder,'/',subs(i).name,'/MEG/Restin/icablpenv/*.nii']));
end

subs(numses==0)=[];

addpath(genpath('/dtu-compute/HCP_dFC/2023/hcp_dfc/src/'))
% parpool(16)

allfreqs = 1:0.5:125;

blp_bands = [ 1.3 4.5 ; 3 9.5 ; 6.3 16.5 ; 12.5 29 ; 22.5 39 ; 30 55 ;  45 82 ; 70 125];% ; 1 150;


for sub = 1:numel(subs)
    for ses = 3:5
        
        d_done = dir(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/eigs_MEG/',subs(sub).name,'/eigs_ses',num2str(ses),'_band*.mat']);
        if numel(d_done)==8
            continue
        end
        
        t0 = tic;
        if ~exist([subs(sub).folder,'/',subs(sub).name,'/MEG/Restin/icablpenv/',subs(sub).name,'_MEG_',num2str(ses),'-Restin_icablpenv_whole.power.dtseries.nii'],'file')
            continue
        end
        data = cifti_read([subs(sub).folder,'/',subs(sub).name,'/MEG/Restin/icablpenv/',subs(sub).name,'_MEG_',num2str(ses),'-Restin_icablpenv_whole.power.dtseries.nii']);
        mkdir(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/eigs_MEG/',subs(sub).name])
        t1 = toc(t0);
        disp(['Loaded data for sub',num2str(sub),' ses',num2str(ses),' in ',num2str(t1/60),' minutes'])
        
%         data = cifti_read('/dtu-compute/HCP_dFC/MEG/data/100307/Restin/icablpenv/100307_MEG_3-Restin_icablpenv_whole.power.dtseries.nii');
        data.times = data.diminfo{2}.seriesStart:data.diminfo{2}.seriesStep:data.diminfo{2}.seriesStep*(data.diminfo{2}.length-1)+data.diminfo{2}.seriesStart;
        
        nSource = size(data.cdata,1);
        ntps = numel(data.times);
        
        for band = 1:size(blp_bands,1)
            t00 = tic;
            if band>1
                freq_old = freq;
                freqtmp = blp_bands(band,1):0.1:blp_bands(band,2);
                freqidx = ismember(freqtmp,allfreqs);
                freq = freqtmp(freqidx);
                [~,alreadydone] = ismember(freq,freq_old);
                
                alreadydone(alreadydone==0)=[];
                firstdone = alreadydone(1);
                lastdone = alreadydone(end);
                alldone = firstdone:lastdone;
                
                
                phase_MEG = phase_MEG(alldone,:,:);
                phase_new = run_all(data,freq(numel(alldone)+1:end));
                phase_MEG = [phase_MEG;phase_new];
                
            else
                freqtmp = blp_bands(band,1):0.1:blp_bands(band,2);
                freqidx = ismember(freqtmp,allfreqs);
                freq = freqtmp(freqidx);
                phase_MEG = run_all(data,freq);
            end
            t2 = toc(t00);
            disp(['Running wavelet for band ',num2str(band),' took ',num2str(t2/60),' minutes'])
            
            bandeigs = nan(ntps,nSource);
            
            A = double([permute(cos(phase_MEG),[3,1,2]),permute(sin(phase_MEG),[3,1,2])]./numel(freq));
            
            for tt = 1:ntps
                [bandeigs(tt,:),~,~] = svds(double(A(:,:,tt)),1);
            end
            t3 = toc(t00);
            disp(['Done running cohmat for band ',num2str(band),' in ',num2str(t3/60),' minutes'])
            save(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/eigs_MEG/',subs(sub).name,'/eigs_ses',num2str(ses),'_band',num2str(band),'.mat'],'bandeigs')
        end
        
        
        t4 = toc(t0);
        disp(['Done with subject ',num2str(sub),' session ',num2str(ses),' in ',num2str(t4/60),' minutes'])
        
    end
end

function phase = run_all(data,freq)
phase = nan(numel(freq),size(data.cdata,2),size(data.cdata,1));

nfreq = numel(freq);
n = linspace(3,10,nfreq);
s = n./(2*pi.*freq);

wavetime = -2:data.diminfo{2}.seriesStep:2;
nWave = length(wavetime);
nData = data.diminfo{2}.length*1; %data.trials
nConv = nWave + nData -1 ;
half_wave = floor(length(wavetime)/2)+1;

% mydata_raw = fft(reshape(squeeze(data.cdata(ind_pickedchannel,:)),1,[]),nConv);
mydata_raw2 = fft(data.cdata',nConv); %anders edit

cmwall = exp(1i*2*pi*freq'*wavetime).*exp(-wavetime.^2./(2*s'.^2));


% for fi=1:nfreq
%     cmwX = fft(cmwall(fi,:),nConv);
%     cmwX = cmwX./max(cmwX);
%     as1 = ifft(cmwX'.*mydata_raw2,nConv);
%     as1 = as1(half_wave:end-half_wave+1,:);
%     phase(fi,:,:) = angle(as1);
% %     disp(['Done with frequency ',num2str(fi),' of ',num2str(nfreq)])
% end

cmwX = fft(cmwall',nConv);
cmwX2(:,1,:) = cmwX./max(cmwX);
as2 = ifft(cmwX2.*mydata_raw2,nConv);
as2 = as2(half_wave:end-half_wave+1,:,:);
phase = angle(as2);
phase = permute(phase,[3,1,2]);


end


% function phase = run_wavelet_ft(data,freq)
% phase = nan(numel(freq),size(data.cdata,2),size(data.cdata,1));
% for source = 1:size(data.cdata,1)
%     [phase(:,:,source),~]=wavelet_ft(data,freq,source);
% end
% end
% 
% function [phase,freq] = wavelet_ft(data,freq,ind_pickedchannel)
% % baseline
% % bl = [-400 -100];
% % ind_bl = dsearchn(EEG.times',bl');
% 
% % wavelet
% % min_freq = 1;
% % max_freq = 80;
% % nfreq = 80;
% % freq = linspace(min_freq,max_freq,nfreq);
% nfreq = numel(freq);
% n = linspace(3,10,nfreq);
% s = n./(2*pi.*freq);
% 
% wavetime = -2:data.diminfo{2}.seriesStep:2;
% nWave = length(wavetime);
% nData = data.diminfo{2}.length*1; %data.trials
% nConv = nWave + nData -1 ;
% half_wave = floor(length(wavetime)/2)+1;
% 
% mydata_raw = fft(reshape(squeeze(data.cdata(ind_pickedchannel,:)),1,[]),nConv);
% 
% % TF transform
% % tf_raw = zeros(nfreq,data.diminfo{2}.length,2); % 1 for power; 2 for phase.
% 
% for fi=1:nfreq
%     cmw = exp(1i*2*pi*freq(fi).*wavetime).*exp(-wavetime.^2./(2*s(fi)^2));
%     cmwX = fft(cmw,nConv);
%     cmwX = cmwX./max(cmwX);
%     
%     % convolve with mydata
%     as = ifft(cmwX.*mydata_raw,nConv);
%     as = as(half_wave:end-half_wave+1);
%     as = reshape(as,data.diminfo{2}.length,1); %data.trials
%     
%     % save power
%     %         temp_power= abs(as).^2;
%     %         tf_raw(fi,:,1) = 10*log10(temp_power/mean(temp_power));
%     
%     % save ITPC
%     %         tf_raw(fi,:,2) = abs(mean(exp(1i*angle(as)),2));
%     phase(fi,:) = angle(as);
% end
% 
% end