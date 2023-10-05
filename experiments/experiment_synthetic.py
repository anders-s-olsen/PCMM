from src.helper_functions import load_data,run_model_reps_and_save_logliks,parse_input_args

# options pertaining to current experiment
options = {}
options['tol'] = 1e-10
options['num_repl_outer'] = 10
options['num_repl_inner'] = 1
options['max_iter'] = 100000
options['outfolder'] = 'data/results/synth_outputs'
options['num_subjects'] = None
options['ACG_rank'] = 'full' #for ACG and MACG
options['data_type'] = 'synthetic'
options['threads'] = 8

def run_experiment(extraoptions={}):
    options.update(extraoptions) #modelname, LR, init controlled in shell script
    print(options)
    ps = [3,10,25]
    Ks = [2,5,10]
    for p in ps:
        for K in Ks:
            print('starting K='+str(K))
            if K>=p:
                continue
            data_train,data_test,data_test2 = load_data(options=options,p=p,K=K)
    
            options['experiment_name'] = '3d_'+options['init']+'_'+str(options['LR'])+'_p'+str(p)+'_K'+str(K)
            run_model_reps_and_save_logliks(data_train=data_train,data_test=data_test,data_test2=data_test2,K=K,options=options)

if __name__=="__main__":
    # options['modelname'] = 'MACG'
    # options['LR'] = 0.1
    # options['init'] = 'unif'
    # run_experiment(options)

    import sys
    print(sys.argv)
    options=parse_input_args(sys.argv)
    print(options)
    run_experiment(extraoptions=options)