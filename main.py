'''
code for the submitted work: "Asynchronous Multi Agent Active Search for Sparse Signals with Region Sensing"

main class file for asynchronous multi agent active search

author: anonymous 

(structure is referenced from parallel Thompson Sampling by:
Kandasamy, K., Krishnamurthy, A., Schneider, J., and
P´oczos, B. Asynchronous parallel Bayesian optimisation
via Thompson sampling. arXiv preprint
arXiv:1705.09236, (2017), GitHub repository: https://github.com/kirthevasank/gp-parallel-ts)
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

def trials(mu, theta2, lmbd, sigma2, EMitr, k, n, max_capital, num_agents, mode, err, alpha, trl):

    rng = np.random.RandomState(trl)
    idx = rng.randint(0,n,size=(k))
    beta = np.zeros((n,1))
    beta[idx,:] = mu+np.sqrt(theta2)*rng.randn(k,1)

# # # #%% SPATS:
    func_class = SPATS(beta, mu, theta2, sigma2, lmbd, EMitr, num_agents, trl)

    worker_manager = WorkerManager(func_caller=func_class, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class)

    beta_hat , history = Bayesian_optimizer(worker_manager, func_class, options).optimize(max_capital)

    est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
    real = (beta>0)

    partial_recovery_rate_spats = np.sum(est==real)/(n)
    correct_spats = 0.0
    if(np.all(est==real)):
        correct_spats = 1.0

        
# #%% RSI:
    func_class2 = RSI(beta, mu, theta2, sigma2, lmbd, EMitr, err, trl)
    

    worker_manager = WorkerManager(func_caller=func_class2, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class2)

    beta_hat , history = Bayesian_optimizer(worker_manager, func_class2, options).optimize(max_capital)

    est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
    real = (beta>0)
    
    partial_recovery_rate_rsi = np.sum(est==real)/(n)
    correct_rsi = 0.0
    if(np.all(est==real)):
        correct_rsi = 1.0  
        
# #%% LATSI:
    func_class = LATSI(beta, mu, theta2, sigma2, lmbd, EMitr, err, num_agents, alpha, trl)
    
    worker_manager = WorkerManager(func_caller=func_class, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class)

    beta_hat , history = Bayesian_optimizer(worker_manager, func_class, options).optimize(max_capital)

    est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
    real = (beta>0)
    
    partial_recovery_rate_LATSI = np.sum(est==real)/(n)
    correct_LATSI = 0.0
    if(np.all(est==real)):
        correct_LATSI = 1.0  


    return [correct_spats, correct_rsi, correct_LATSI, partial_recovery_rate_spats,partial_recovery_rate_rsi, partial_recovery_rate_LATSI]
    


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
    k = 1 # sparsity rate
    num_trials = 5 # number of trials
    n = 8 # length n of vector beta
    rangeT = np.array([2,4]) # list on number of measurements T
    err = 0.5 # hyperparameter for RSI algorithm
    alpha = 1 # hyper parameter for LATSI algorithm
    
    num_agents = np.array([1,2]) # list on number of agents
    mode = 'asy' #alternatively 'syn' defines synchronous vs. asynchronous parallelisation. we focus on 'asy' in this paper

    full_recovery_rate = np.zeros((num_agents.shape[0], rangeT.shape[0],num_trials,3)) # percentage of results where we fully recover a vector beta
    partial_recovery_rate = np.zeros((num_agents.shape[0], rangeT.shape[0],num_trials,3))  # percentage of estimating correct entries

    TT = 0
    aid = 0
    
    for agents in num_agents:
        TT = 0
        
        for T in rangeT:
            print('agents: %d T=%d'%(agents,T))
            if(TT==0):
                schftseed = 0
            else:
                schftseed = rangeT[TT-1] * (num_trials+1)
            result = Parallel(n_jobs=5, prefer='processes')(delayed(trials)(mu, theta2, lmbd, sigma2, EMitr, k, n, T, agents, mode, err, alpha, schftseed+T*trl) for trl in range(num_trials))
            res = np.array(result)
            full_recovery_rate[aid,TT,:,0] = res[:,0]#SPATS
            full_recovery_rate[aid,TT,:,1] = res[:,1]#RSI
            full_recovery_rate[aid,TT,:,2] = res[:,2]#LATSI
            partial_recovery_rate[aid,TT,:,0] = res[:,3]#SPATS
            partial_recovery_rate[aid,TT,:,1] = res[:,4]#RSI
            partial_recovery_rate[aid,TT,:,2] = res[:,5]#LATSI
            TT += 1
        aid += 1
    # print('recovery ',full_recovery_rate[:,:,:,0])
    savepath = 'results/'
    filename = ('results.pkl')
    
    with open(os.path.join(savepath,filename),'wb') as f:
        pickle.dump([rangeT,full_recovery_rate,partial_recovery_rate],f)

    print('saved!')
  
    LATSI_recovery = np.zeros((num_agents.shape[0], rangeT.shape[0], num_trials, 2))
    SPATS_recovery = np.zeros((num_agents.shape[0], rangeT.shape[0], num_trials, 2))
    RSI_recovery = np.zeros((num_agents.shape[0], rangeT.shape[0], num_trials, 2))

    f_std_err_LATSI = np.zeros((num_agents.shape[0], rangeT.shape[0]))
    p_std_err_LATSI = np.zeros((num_agents.shape[0], rangeT.shape[0]))


    f_std_err_SPATS = np.zeros((num_agents.shape[0], rangeT.shape[0]))
    p_std_err_SPATS = np.zeros((num_agents.shape[0], rangeT.shape[0]))

    f_std_err_RSI = np.zeros((num_agents.shape[0], rangeT.shape[0]))
    p_std_err_RSI = np.zeros((num_agents.shape[0], rangeT.shape[0]))

    with open('results/results.pkl', 'rb') as f:
        data = pickle.load(f)

    RSI_recovery[0,:,:,0] = data[1][0,:,:,1]
    f_std_err_RSI[0,:] = stats.sem(RSI_recovery[0,:,:,0], axis=1)
    RSI_recovery[0,:,:,1] = data[2][0,:,:,1]
    p_std_err_RSI[0,:] = stats.sem(RSI_recovery[0,:,:,1], axis=1)

    SPATS_recovery[0,:,:,0] = data[1][0,:,:,0]
    f_std_err_SPATS[0,:] = stats.sem(SPATS_recovery[0,:,:,0], axis=1)
    SPATS_recovery[0,:,:,1] = data[2][0,:,:,0]
    p_std_err_SPATS[0,:] = stats.sem(SPATS_recovery[0,:,:,1], axis=1)

    LATSI_recovery[0,:,:,0] = data[1][0,:,:,2]
    f_std_err_LATSI[0,:] = stats.sem(LATSI_recovery[0,:,:,0], axis=1)
    LATSI_recovery[0,:,:,1] = data[2][0,:,:,2]
    p_std_err_LATSI[0,:] = stats.sem(LATSI_recovery[0,:,:,1], axis=1)

    SPATScolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
    LATSIcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
    RSIcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))

    marker = ["o","d","s","*"]
    plt.figure(figsize = (8,6))
    for aid,_ in enumerate(num_agents):
        sb.tsplot(time=rangeT,data=np.mean(LATSI_recovery[aid,:,:,0], axis=1), color=next(LATSIcolor), condition='LATSI-'+str(num_agents[aid]), linestyle='dashed')
        sb.tsplot(time=rangeT,data=np.mean(SPATS_recovery[aid,:,:,0], axis=1), color=next(SPATScolor), condition='SPATS-'+str(num_agents[aid]), linestyle='solid')
        sb.tsplot(time=rangeT,data=np.mean(RSI_recovery[aid,:,:,0], axis=1), color=next(RSIcolor), condition='RSI-'+str(num_agents[aid]), linestyle='dashdot')
    SPATSfillcolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
    LATSIfillcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
    RSIfillcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))
    for aid,_ in enumerate(num_agents):
        plt.fill_between(rangeT, np.mean(LATSI_recovery[aid,:,:,0],axis=1)+f_std_err_LATSI[aid,:], np.mean(LATSI_recovery[aid,:,:,0],axis=1)-f_std_err_LATSI[aid,:], color=next(LATSIfillcolor), alpha=0.5)
        plt.fill_between(rangeT, np.mean(SPATS_recovery[aid,:,:,0],axis=1)+f_std_err_SPATS[aid,:], np.mean(SPATS_recovery[aid,:,:,0],axis=1)-f_std_err_SPATS[aid,:], color=next(SPATSfillcolor), alpha=0.5)
        plt.fill_between(rangeT, np.mean(RSI_recovery[aid,:,:,0],axis=1)+f_std_err_RSI[aid,:], np.mean(RSI_recovery[aid,:,:,0],axis=1)-f_std_err_RSI[aid,:], color=next(RSIfillcolor), alpha=0.5)
        
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
        #sb.tsplot(time=rangeT_less/agents,data=np.mean(LTR_recovery[aid,:,:,0], axis=1), color=next(LTRcolor), condition='LATSI-'+str(num_agents[aid]), linestyle='dashed')
        sb.tsplot(time=rangeT/agents,data=np.mean(LATSI_recovery[aid,:,:,0], axis=1), color=next(LATSIcolor), condition='LATSI-'+str(num_agents[aid]), linestyle='dashed')
        sb.tsplot(time=rangeT/agents,data=np.mean(SPATS_recovery[aid,:,:,0], axis=1), color=next(SPATScolor), condition='SPATS-'+str(num_agents[aid]), linestyle='solid')
        sb.tsplot(time=rangeT/agents,data=np.mean(RSI_recovery[aid,:,:,0], axis=1), color=next(RSIcolor), condition='RSI-'+str(num_agents[aid]), linestyle='dashdot')
    SPATSfillcolor=iter(cm.winter(np.linspace(0,1,num_agents.shape[0])))
    LATSIfillcolor=iter(cm.autumn(np.linspace(0,1,num_agents.shape[0])))
    RSIfillcolor=iter(cm.summer(np.linspace(0,1,num_agents.shape[0])))
    for aid, agents in enumerate(num_agents):
        plt.fill_between(rangeT/agents, np.mean(LATSI_recovery[aid,:,:,0],axis=1)+f_std_err_LATSI[aid,:], np.mean(LATSI_recovery[aid,:,:,0],axis=1)-f_std_err_LATSI[aid,:], color=next(LATSIfillcolor), alpha=0.5)
        plt.fill_between(rangeT/agents, np.mean(SPATS_recovery[aid,:,:,0],axis=1)+f_std_err_SPATS[aid,:], np.mean(SPATS_recovery[aid,:,:,0],axis=1)-f_std_err_SPATS[aid,:], color=next(SPATSfillcolor), alpha=0.5)
        plt.fill_between(rangeT/agents, np.mean(RSI_recovery[aid,:,:,0],axis=1)+f_std_err_RSI[aid,:], np.mean(RSI_recovery[aid,:,:,0],axis=1)-f_std_err_RSI[aid,:], color=next(RSIfillcolor), alpha=0.5)

    plt.legend()
    plt.xlabel("time (T/g)",fontsize = 18)
    plt.ylabel("full recovery rate",fontsize = 18)
    plt.xlim(0,n)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title("k=%d"%k, fontsize=18)
    plt.savefig('results/Tbyagents_full_recovery_agents_%s_k_%d_n_%d_trials_%d.pdf'%(str(num_agents),k,n,num_trials))
    plt.show()

