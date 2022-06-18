function Vid_MeanScore(ResDir,MaxVideoNums)
    if nargin < 2
        MaxVideoNums = Inf;      
    end   
    if nargin < 1
        ResDir = fullfile('/home/kao/D/DataSet/UAV2-TE/Results/Results_Oth/');
    end

    scoreName = 'Scores';
    scoreDir = [ResDir scoreName filesep];
    outDir   = [ResDir filesep];

    d = dir(scoreDir);
    methodFiles = {d([d.isdir]).name}';
	methodFiles = methodFiles(3:end);
    methodNum = length(methodFiles);

    meanS = struct;
    for i = 1:methodNum
        iscoreDir = [scoreDir methodFiles{i} filesep];

        d = dir([iscoreDir, '*.mat']);
        iscoreFiles = {d(~[d.isdir]).name}';
        iscoreNum = min(length(iscoreFiles),MaxVideoNums);

        iScores = struct;
        ims = [];
        for idx_s = 1:iscoreNum
            ifilename = iscoreFiles{idx_s};
            iscore = load([iscoreDir iscoreFiles{idx_s}]);
            iscore = iscore.iscore;

            iScores(idx_s).name = ifilename;
            iScores(idx_s).scores = iscore;

            tmp = ~isnan(sum(iscore,2));
%             ims = [ims;mean(iscore(tmp,:))];
            ims = [ims;iscore(tmp,:)];

% 			iscore(isnan(iscore)) = 0;
%             ims = [ims;mean(iscore)];

            iScores(idx_s).ims = mean(iscore(tmp,:));
        end

        tmp = ~isnan(sum(ims,2));
        tms = mean(ims(tmp,:));

        meanS(i).name= methodFiles{i};
        meanS(i).AUC_S  = tms(1);
        meanS(i).NSS    = tms(2);
        meanS(i).AUC_J  = tms(3);
        meanS(i).AUC_B  = tms(4);
        meanS(i).KL     = tms(5);
        meanS(i).SIM    = tms(6);
        meanS(i).CC     = tms(7);
        meanS(i).scores = iScores;
    end
    save([outDir 'mean' scoreName '.mat'],'meanS');

end
