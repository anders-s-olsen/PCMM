from src.helper_functions import load_data,run_model_reps_and_save_logliks,parse_input_args

def run_experiment(extraoptions={}):
    # options pertaining to current experiment
    options = {}
    options['tol'] = 1e-8
    options['num_repl_outer'] = 10
    options['num_repl_inner'] = 1
    options['max_iter'] = 100000
    options['num_subjects'] = 100
    options['ACG_rank'] = 25 #for ACG and MACG
    options['outfolder'] = 'data/results/116_GSR_vs_noGSR_outputs'
    options.update(extraoptions) #modelname, LR, init, GSR controlled in shell script
    
    ## load data, only the first 100 subjects (each with 1200 data points)
    if options['GSR']==0:
        options['data_type'] = 'fMRI_SchaeferTian116'
    elif options['GSR']==1:
        options['data_type'] = 'fMRI_SchaeferTian116_GSR'
    data_train,data_test,data_test2 = load_data(options=options)
    p = data_train.shape[1]
    
    for K in [2,7]:
        options['experiment_name'] = '116_'+options['init']+'_'+str(options['LR'])+'_p'+str(p)+'_K'+str(K)
        run_model_reps_and_save_logliks(data_train=data_train,data_test=data_test,data_test2=data_test2,K=K,options=options)


if __name__=="__main__":
    # options['modelname'] = 'ACG_lowrank'
    # options['LR'] = float(0)
    # options['init'] = 'unif'
    # run_experiment(options)

    import sys
    options=parse_input_args(sys.argv)
    run_experiment(options)