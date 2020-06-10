'''
Code for the following publication: 
Ramina Ghods, Arundhati Banerjee, Jeff Schneider, ``Asynchronous Multi Agent Active Search 
for Sparse Signals with Region Sensing", 
2020 international conference on machine learning (ICML) (submitted)

(c) Feb 9, 2020: Ramina Ghods (rghods@cs.cmu.edu), Arundhati Banerjee (arundhat@andrew.cmu.edu)
Please do not distribute. The code will become public upon acceptance of the paper.

main class file for asynchronous multi agent active search

'''

from SPATS import SPATS
from RSI import RSI
from LATSI import LATSI
from bayesian_optimization import Bayesian_optimizer
from worker_manager import WorkerManager
from argparse import Namespace 
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from joblib import Parallel, delayed
import seaborn as sb
from matplotlib.pyplot import cm
from scipy import stats

def trials(length, width, mu, theta2, lmbd, sigma2, EMitr, k, n, max_capital, num_agents, mode, err, alpha, trl):

    n = length*width
    rng = np.random.RandomState(trl)
    idx = rng.randint(0,n,size=(k))
    beta = np.zeros((n,1))
    beta[idx,:] = mu+np.sqrt(theta2)*rng.randn(k,1)

# # # #%% SPATS:
    func_class = SPATS(length, width, beta, mu, theta2, sigma2, lmbd, EMitr, num_agents, trl)

    worker_manager = WorkerManager(func_caller=func_class, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class)

    beta_hats = Bayesian_optimizer(trl, worker_manager, func_class, options).optimize(max_capital)

    full_recovery_rate_spats = []
    partial_recovery_rate_spats = []

    for i in range(max_capital):
        beta_hat = beta_hats[i]

        est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
        real = (beta>0)
    
        partial_recovery_rate_spats.append(np.sum(est==real)/(n))
        correct_spats = 0.0
        if(np.all(est==real)):
            correct_spats = 1.0  
        full_recovery_rate_spats.append(correct_spats)

    

# #%% RSI:
    func_class2 = RSI(length, width, beta, mu, theta2, sigma2, lmbd, EMitr, err, trl)
    

    worker_manager = WorkerManager(func_caller=func_class2, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class2)

    beta_hats = Bayesian_optimizer(trl, worker_manager, func_class2, options).optimize(max_capital)

    full_recovery_rate_rsi = []
    partial_recovery_rate_rsi = []

    for i in range(max_capital):
        beta_hat = beta_hats[i]

        est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
        real = (beta>0)
    
        partial_recovery_rate_rsi.append(np.sum(est==real)/(n))
        correct_rsi = 0.0
        if(np.all(est==real)):
            correct_rsi = 1.0  
        full_recovery_rate_rsi.append(correct_rsi)

# #%% LATSI:
    func_class = LATSI(length, width, beta, mu, theta2, sigma2, lmbd, EMitr, err, num_agents, alpha, trl)
    
    worker_manager = WorkerManager(func_caller=func_class, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class)

    beta_hats = Bayesian_optimizer(trl, worker_manager, func_class, options).optimize(max_capital)

    full_recovery_rate_latsi = []
    partial_recovery_rate_latsi = []

    for i in range(max_capital):
        beta_hat = beta_hats[i]

        est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
        real = (beta>0)
    
        partial_recovery_rate_latsi.append(np.sum(est==real)/(n))
        correct_LATSI = 0.0
        if(np.all(est==real)):
            correct_LATSI = 1.0  
        full_recovery_rate_latsi.append(correct_LATSI)
    
    
    
    return [full_recovery_rate_spats, full_recovery_rate_rsi, full_recovery_rate_latsi, partial_recovery_rate_spats,partial_recovery_rate_rsi, partial_recovery_rate_latsi]
    


if __name__ == "__main__":
    '''
    Input Arguments:
      - func: The function to be optimised.
      - num_agents:
      - max_capital: The maximum capital for optimisation.
      - options: A namespace which gives other options.
     
      - #true_opt_pt, true_opt_val: The true optimum point and value (if known). Mostly for
          experimenting with synthetic problems.
      
    Returns: (gpb_opt_pt, gpb_opt_val, history)
      - gpb_opt_pt, gpb_opt_val: The optimum point and value.
      - history: A namespace which contains a history of all the previous queries.
    '''

    print('start process')
    mu = 16 # signal intensity to create nonzero entries of vector beta, this parameter is not used for estimation
    theta2 = 4 # signal variance to create nonzero entries of vector beta
    lmbd = 1 # Laplace hyper parameter lmbd = sqrt(eta) where eta is introduced in the paper
    sigma2 = 1 # noise variance on observations
    EMitr = 30 # number of iterations for the Expectation-Maximization estimator
    k_arr = np.array([1,5]) # sparsity rate
    num_trials = 50 # number of trials
    # 2D vector beta of dimension length X width
    length = 16
    width = 16
    n = length*width # flattened length n of vector beta
    T = 128 # number of measurements T
    err = 0.5 # hyperparameter for RSI algorithm
    alpha = 1 # hyper parameter for LATSI algorithm
    
    num_agents = np.array([1,2,4,8]) # list on number of agents
    mode = 'asy' #alternatively 'syn' defines synchronous vs. asynchronous parallelisation. we focus on 'asy' in this paper

    full_recovery_rate = np.zeros((num_agents.shape[0], T, num_trials,3)) # percentage of results where we fully recover a vector beta
    partial_recovery_rate = np.zeros((num_agents.shape[0], T, num_trials,3))  # percentage of estimating correct entries

    
    for k in k_arr:
        aid = 0
        
        for agents in num_agents:
            print('agents: %d T=%d'%(agents,T))
            schftseed = T * (num_trials+1)
            result = Parallel(n_jobs=25, prefer='processes')(delayed(trials)(length, width, mu, theta2, lmbd, sigma2, EMitr, k, n, T, agents, mode, err, alpha, schftseed+T*trl) for trl in range(num_trials))
            #print(result)
            res = np.array(result)
            full_recovery_rate[aid,:,:,0] = np.stack(res[:,0]).T#SPATS
            full_recovery_rate[aid,:,:,1] = np.stack(res[:,1]).T#RSI
            full_recovery_rate[aid,:,:,2] = np.stack(res[:,2]).T#LATSI
            
            partial_recovery_rate[aid,:,:,0] = np.stack(res[:,3]).T#SPATS
            partial_recovery_rate[aid,:,:,1] = np.stack(res[:,4]).T#RSI
            partial_recovery_rate[aid,:,:,2] = np.stack(res[:,5]).T#LATSI
            
            aid += 1
        # print('recovery ',full_recovery_rate[:,:,:,0])
        savepath = 'results/'
        filename = 'results.pkl'
        
        with open(os.path.join(savepath,filename),'wb') as f:
            pickle.dump([T,full_recovery_rate,partial_recovery_rate],f)

        print('saved!')
      
        LATSI_recovery = np.zeros((num_agents.shape[0], T, num_trials, 2))
        SPATS_recovery = np.zeros((num_agents.shape[0], T, num_trials, 2))
        RSI_recovery = np.zeros((num_agents.shape[0], T, num_trials, 2))

        f_std_err_LATSI = np.zeros((num_agents.shape[0], T))
        p_std_err_LATSI = np.zeros((num_agents.shape[0], T))


        f_std_err_SPATS = np.zeros((num_agents.shape[0], T))
        p_std_err_SPATS = np.zeros((num_agents.shape[0], T))

        f_std_err_RSI = np.zeros((num_agents.shape[0], T))
        p_std_err_RSI = np.zeros((num_agents.shape[0], T))

        with open('results/results.pkl', 'rb') as f:
            data = pickle.load(f)

        for i in range(num_agents.shape[0]):
            RSI_recovery[i,:,:,0] = data[1][i,:,:,1]
            f_std_err_RSI[i,:] = stats.sem(RSI_recovery[i,:,:,0], axis=1)
            RSI_recovery[i,:,:,1] = data[2][i,:,:,1]
            p_std_err_RSI[i,:] = stats.sem(RSI_recovery[i,:,:,1], axis=1)

            SPATS_recovery[i,:,:,0] = data[1][i,:,:,0]
            f_std_err_SPATS[i,:] = stats.sem(SPATS_recovery[i,:,:,0], axis=1)
            SPATS_recovery[i,:,:,1] = data[2][i,:,:,0]
            p_std_err_SPATS[i,:] = stats.sem(SPATS_recovery[i,:,:,1], axis=1)

            LATSI_recovery[i,:,:,0] = data[1][i,:,:,2]
            f_std_err_LATSI[i,:] = stats.sem(LATSI_recovery[i,:,:,0], axis=1)
            LATSI_recovery[i,:,:,1] = data[2][i,:,:,2]
            p_std_err_LATSI[i,:] = stats.sem(LATSI_recovery[i,:,:,1], axis=1)

        SPATScolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
        LATSIcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
        RSIcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))

        marker = ["o","d","s","*"]
        plt.figure(figsize = (8,6))
        for aid,_ in enumerate(num_agents):
            sb.tsplot(time=np.arange(T),data=np.mean(LATSI_recovery[aid,:,:,0], axis=1), color=next(LATSIcolor), condition='LATSI-'+str(num_agents[aid]), linestyle='dashed')
            sb.tsplot(time=np.arange(T),data=np.mean(SPATS_recovery[aid,:,:,0], axis=1), color=next(SPATScolor), condition='SPATS-'+str(num_agents[aid]), linestyle='solid')
            sb.tsplot(time=np.arange(T),data=np.mean(RSI_recovery[aid,:,:,0], axis=1), color=next(RSIcolor), condition='RSI-'+str(num_agents[aid]), linestyle='dashdot')
        SPATSfillcolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
        LATSIfillcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
        RSIfillcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))
        for aid,_ in enumerate(num_agents):
            plt.fill_between(np.arange(T), np.mean(LATSI_recovery[aid,:,:,0],axis=1)+f_std_err_LATSI[aid,:], np.mean(LATSI_recovery[aid,:,:,0],axis=1)-f_std_err_LATSI[aid,:], color=next(LATSIfillcolor), alpha=0.5)
            plt.fill_between(np.arange(T), np.mean(SPATS_recovery[aid,:,:,0],axis=1)+f_std_err_SPATS[aid,:], np.mean(SPATS_recovery[aid,:,:,0],axis=1)-f_std_err_SPATS[aid,:], color=next(SPATSfillcolor), alpha=0.5)
            plt.fill_between(np.arange(T), np.mean(RSI_recovery[aid,:,:,0],axis=1)+f_std_err_RSI[aid,:], np.mean(RSI_recovery[aid,:,:,0],axis=1)-f_std_err_RSI[aid,:], color=next(RSIfillcolor), alpha=0.5)
            
        plt.legend()
        plt.xlabel("number of measurements (T)",fontsize = 18)
        plt.ylabel("full recovery rate",fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.title("k=%d"%k, fontsize=18)
        plt.savefig('results/T_full_recovery_agents_%s_k_%d_n_%d_trials_%d.pdf'%(str(num_agents),k,n,num_trials))
        plt.show()

        SPATScolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
        LATSIcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
        RSIcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))

        marker = ["o","d","s","*"]
        plt.figure(figsize = (8,6))
        for aid, agents in enumerate(num_agents):
            sb.tsplot(time=np.arange(T)/agents,data=np.mean(LATSI_recovery[aid,:,:,0], axis=1), color=next(LATSIcolor), condition='LATSI-'+str(num_agents[aid]), linestyle='dashed')
            sb.tsplot(time=np.arange(T)/agents,data=np.mean(SPATS_recovery[aid,:,:,0], axis=1), color=next(SPATScolor), condition='SPATS-'+str(num_agents[aid]), linestyle='solid')
            sb.tsplot(time=np.arange(T)/agents,data=np.mean(RSI_recovery[aid,:,:,0], axis=1), color=next(RSIcolor), condition='RSI-'+str(num_agents[aid]), linestyle='dashdot')
        SPATSfillcolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
        LATSIfillcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
        RSIfillcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))
        for aid, agents in enumerate(num_agents):
            plt.fill_between(np.arange(T)/agents, np.mean(LATSI_recovery[aid,:,:,0],axis=1)+f_std_err_LATSI[aid,:], np.mean(LATSI_recovery[aid,:,:,0],axis=1)-f_std_err_LATSI[aid,:], color=next(LATSIfillcolor), alpha=0.5)
            plt.fill_between(np.arange(T)/agents, np.mean(SPATS_recovery[aid,:,:,0],axis=1)+f_std_err_SPATS[aid,:], np.mean(SPATS_recovery[aid,:,:,0],axis=1)-f_std_err_SPATS[aid,:], color=next(SPATSfillcolor), alpha=0.5)
            plt.fill_between(np.arange(T)/agents, np.mean(RSI_recovery[aid,:,:,0],axis=1)+f_std_err_RSI[aid,:], np.mean(RSI_recovery[aid,:,:,0],axis=1)-f_std_err_RSI[aid,:], color=next(RSIfillcolor), alpha=0.5)

        plt.legend()
        plt.xlabel("time (T/g)",fontsize = 18)
        plt.ylabel("full recovery rate",fontsize = 18)
        plt.xlim(0,n)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.title("k=%d"%k, fontsize=18)
        plt.savefig('results/Tbyagents_full_recovery_agents_%s_k_%d_n_%d_trials_%d.pdf'%(str(num_agents),k,n,num_trials))
        plt.show()

