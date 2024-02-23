from simulation import MASsimulation
import numpy as np
from utils import *
import concurrent.futures
import time
from synthetis import ControlEstimationSynthesis
import pickle
import os
from analysis import Analysis

def run_simulation(args):
    sim = MASsimulation(args)
    # The constructor of MASsimulation starts the simulation automatically, no need to call run_sim here
    return sim.eval.get_results()

def count_completed_tasks(futures):   
    return sum(1 for future in futures if future.done())

def mcmc_simulatoin(num_simulations = 100, args = None, ctrl_type  = CtrlTypes.LQROutputFeedback):
    
    max_concurrent_processes = 10  # Define the maximum number of concurrent processes    
    args_list = []
    for _ in range(num_simulations):
        args['ctrl_type'] = ctrl_type        
        args_list.append(args)
    obj = MASsimulation(args)
    # Create a list to store the evaluation results for each run    
    results_list = []    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_processes) as executor:        
        futures = [executor.submit(run_simulation, args) for args in args_list]
        while count_completed_tasks(futures) < num_simulations:
            print(f"Completed tasks: {count_completed_tasks(futures)} / {num_simulations}")
            time.sleep(1)          
        for future in futures:
            results_list.append(future.result())

    ########## Collect data from simulations ##############
    stage_costs = [result['stage_cost'] for result in results_list]
    np_stage_costs = np.stack(stage_costs).squeeze()
    avg_stage_costs = np.mean(np_stage_costs, axis=0)
    
    trajs = [result['trajs'] for result in results_list]    
    np_trajs = np.stack(trajs).squeeze()
    avg_trajs = np.mean(np_trajs, axis=0)
    
    est_trajs = [result['est_trajs'] for result in results_list]
    np_est_trajs = np.stack(est_trajs).squeeze()
   
    avg_est_trajs = np.mean(np_est_trajs, axis=0)
    
    result = {'stage_cost' : np_stage_costs, 
              'avg_trajs' : avg_trajs}
    return result
    


def get_fully_connected_args(args):
    new_args = args.copy()
    new_args['c'] =  np.ones([args['N'],args['N']]) # adjencency matrix 
    new_args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix             
    new_args['gain_file_name'] = 'fully' + str(args['N'])
    return new_args
    
if __name__ == "__main__":
    
    lqr_results = []
    lqg_results = []
    opt_results = []
    sub_results = []
    comlqg_results = []
    comlqg_5_results = []
    fullyconnected_synthesis_list = []
    partial_synthesis_list = [] 

    num_simulations = 5  # Define the number of parallel simulations
    args = {}        
    args['sim_n_step'] = 200
    args['n'] = 4
    args['p'] = 2
    args['Ts'] = 0.1   
    args['ctrl_type'] = 0
    args['gamma'] = None
    w_std = 0.1    
    v_std = 0.1   
    
    num_agent_list = [5,10]
    gamma_list = [3,20]
    for idx, N_agent in enumerate(num_agent_list):
        args['gamma'] = gamma_list[idx]
        args['N'] = N_agent
        args['w_std'] = w_std  
        args['v_std'] = np.ones([N_agent,1])*v_std
        args['v_std'][0] = args['v_std'][0]*10            
        args['c'] = get_chain_adj_mtx(N_agent)         
        args['L'] = get_laplacian_mtx(args['c']) 
        args['Q'] = np.kron(args['L'], np.eye(args['n'])) # np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])    
        args['R'] = np.eye(N_agent)
        

        fullyconnected_args = get_fully_connected_args(args.copy())
        fullyconnected_args['gain_file_name'] = 'lqg' + str(args['N'])
        fullyconnected_args['ctrl_type'] = CtrlTypes.LQGFeedback
        fullyconnected_synthesis = ControlEstimationSynthesis(fullyconnected_args)
        fullyconnected_synthesis_list.append(fullyconnected_synthesis)


        sub_args = args.copy()
        sub_args['gain_file_name'] = 'sub' + str(args['N'])
        sub_args['ctrl_type'] = CtrlTypes.SubOutpFeedback
        sub_synthesis = ControlEstimationSynthesis(sub_args)
        
        comglqg_args = args.copy()
        comglqg_args['gain_file_name'] = 'comlqg' + str(args['N'])
        comglqg_args['ctrl_type'] = CtrlTypes.COMLQG
        comglqg_args['gamma'] = args['gamma'] 
        comglqg_synthesis = ControlEstimationSynthesis(comglqg_args)


        partial_args = args.copy()
        partial_args['gain_file_name'] = 'ctrlest' +str(args['N'])    
        partial_args['ctrl_type'] = CtrlTypes.CtrlEstFeedback
        partial_synthesis = ControlEstimationSynthesis(partial_args)
        partial_synthesis_list.append(partial_synthesis)
        
        # LQROutputFeedback = 0        
        # SubOutpFeedback = 1 
        # CtrlEstFeedback = 2 
        # LQGFeedback 3   
        
        lqg_result = mcmc_simulatoin(num_simulations, fullyconnected_args,CtrlTypes.LQGFeedback)
        lqg_results.append(lqg_result) 
        print('LQR with {} agents Done'.format(N_agent))

        
        sub_result = mcmc_simulatoin(num_simulations, sub_args,CtrlTypes.SubOutpFeedback)
        sub_results.append(sub_result)  
        print('Suboptimal with {} agents Done'.format(N_agent))

        comlqg_result = mcmc_simulatoin(num_simulations, comglqg_args,CtrlTypes.COMLQG)
        comlqg_results.append(comlqg_result)  
        print('Comlqg with {} agents Done'.format(N_agent))
    

        opt_result = mcmc_simulatoin(num_simulations, partial_args,CtrlTypes.CtrlEstFeedback)  
        opt_results.append(opt_result)
        
        print('Opt with {} agents Done'.format(N_agent))
        print('MCMCs with {} agents Done'.format(N_agent))
               
        
        
        

    save_data = {}
    save_data['args'] = args
    save_data['num_agent_list'] = num_agent_list        
    save_data['lqg_results'] = lqg_results
    save_data['opt_results'] = opt_results
    save_data['sub_results'] = sub_results
    save_data['comlqg_results'] = comlqg_results    
    
    
    save_file_name = 'mcmc_experiment_string' +str(10) + str('.pkl')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')  
    file_path = os.path.join(data_dir, save_file_name) 
    if os.path.exists(file_path):
        file_path = file_path.split('.pkl')[0]+'_copy.pkl'
    with open(file_path, 'wb') as file:
         pickle.dump(save_data,file)
    
    
    # plot_comparison_result(lqg_results, sub_results, opt_results, comlqg_results)
    analysis_v1 = Analysis(save_file_name)    
    analysis_v1.draw_plot()
