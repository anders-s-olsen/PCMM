subjects = table2array(readtable('paper/data/255unrelatedsubjectsIDs.txt'));

for i = 1:length(subjects)
    subID = subjects(i);
    topFolder = ['test/' num2str(subID)];
    destPath = fullfile('paper/data/raw', num2str(subID),'fMRI');
    % find every file for this subject in the top folder ending with .dtseries.nii
    files = dir(fullfile(topFolder, '**', '*.dtseries.nii'));
    for j = 1:length(files)
        filePath = fullfile(files(j).folder, files(j).name);
        movefile(filePath, destPath);
        %print the command instead of executing it
        % fprintf('movefile(''%s'', ''%s'');\n', filePath, destPath);
    end
end