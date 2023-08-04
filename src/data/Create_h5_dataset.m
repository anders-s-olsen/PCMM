clear
maxNumCompThreads('automatic')
PED = 'RL';
allsessions = dir(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full/*rfMRI_REST*',PED,'*.mat']);
%% not enough data
% lengths = nan(numel(allsessions),1);
% for i = 1:numel(allsessions)
%     try
%     data = load([allsessions(i).folder,'/',allsessions(i).name]);
%     
%     lengths(i) = size(data.dopt,1);
%     disp([num2str(i),' of ',num2str(numel(allsessions))]);
%     catch
%         lengths(i)=0;
%     end
% end
% save('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/lengths','lengths');
load('/dtu-compute/HCP_dFC/2023/hcp_dfc/data/lengths','lengths');
toosmall = lengths~=1200;
allsessions(toosmall)=[];
%% locate data, remove subjects with only 1 session

for i = 1:numel(allsessions)
    session_subject{i} = allsessions(i).name(1:6);
end
[gc,gr] = groupcounts(session_subject');
only1session = gr(gc==1);

for i = 1:numel(only1session)
    remove_session(i) =find(strcmp(session_subject,only1session(i)));
end
allsessions(remove_session)=[];

%% create RL dataset
tic
for spl = 1:2
    split = num2str(spl);
    datadir = allsessions(~cellfun(@isempty,regexp({allsessions.name},[split,'_',PED],'once')));
    h5create(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full_',PED,split,'.h5'],'/Dataset',[Inf,91282],'Chunksize',[1,91282]);
%     h5create(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full_',PED,split,'.h5'],'/Length',[numel(datadir),1]);
%         h5create(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454_',PED,split,'.h5'],'/Dataset',[Inf,454],'Chunksize',[1,454]);
%         h5create(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454_',PED,split,'.h5'],'/Length2',[numel(datadir),1]);
    start = 1;
    for i = 1:numel(datadir)
        data = load([datadir(i).folder,'/',datadir(i).name]);
%                 data_atlas = load(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454/',datadir(i).name]);
        [n,~] = size(data.dopt);
        h5write(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full_',PED,split,'.h5'],'/Dataset',data.dopt,[start,1],[n,91282]);
%         h5write(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_full_',PED,split,'.h5'],'/Length',n,[i,1],[1,1]);
%                 h5write(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454_',PED,split,'.h5'],'/Dataset',data_atlas.dopt,[start,1],[n,454]);
%                 h5write(['/dtu-compute/HCP_dFC/2023/hcp_dfc/data/processed/fMRI_SchaeferTian454_',PED,split,'.h5'],'/Length',n,[i,1],[1,1]);
        disp(['Done with ',num2str(i),' of ',num2str(numel(datadir))])
        start = start + n;
        toc
    end
end
