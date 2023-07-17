p=454;
name = 'Watson'
for ini = {'unif','++','dc'}
    for K = [2,5,10]
        expname = ['454_',ini{1},'_-0.0_p',num2str(p),'_K',num2str(K)];
        for rep = 1:10
            if K==2 && rep==1
                continue
            end

            a = readtable(['experiments/454_outputs/',name,'_',expname,'_traintestlikelihood_r',num2str(rep),'.csv']);
            a.Var1(2) = a.Var2(1);
            a.Var2 = [];
            writetable(a,['experiments/454_outputs/',name,'_',expname,'_traintestlikelihood_r',num2str(rep-1),'.csv'],'WriteVariableNames',false)
        end
    end
end