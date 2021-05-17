'''
Code for the following paper:

Ramina Ghods, Arundhati Banerjee, Jeff Schneider, ``Decentralized Multi-Agent Active Search for Sparse Signals",
2021 Conference On Uncertainty in Artificial Intelligence (UAI)

(c) Ramina Ghods(rghods@cs.cmu.edu), Arundhati Banerjee(arundhat@andrew.cmu.edu)

In this file, we are coding the RSI algorithm from reference:
Ma, Y., Garnett, R., and Schneider, J. Active search for
sparse signals with region sensing. In Thirty-First AAAI
Conference on Artificial Intelligence, 2017.

We use this code to compare RSI algorithm from the aforementioned reference to our
proposed SPATS and LATSI algorithms.

'''

import numpy as np
import math
import pickle as pkl
import scipy.stats as ss

import copy
import os

class RSI(object):

    def __init__(self, beta, mu, theta2, sigma2, lmbd, EMitr, err, trl):
        self.beta = beta
        self.mu = mu
        self.theta2 = theta2
        self.sigma2 = sigma2
        self.lmbd = lmbd
        self.EMitr = EMitr
        self.trl = trl

        self.n = beta.shape[0]
        self.L = int(self.n/2)
        self.M = int(self.n / self.L)

        self.rng = np.random.RandomState(trl)

        self.gamma = self.rng.rand(self.M)
        self.B = self.rng.rand(self.L,self.L)
        self.Sig0 = np.kron(np.diag(self.gamma),self.B)
        self.err = err
#
#        print('init RSI')


    def sample_from_prior(self, num_samples):

        beta_tilde = np.zeros((num_samples, self.n, 1))

        for idx in range(num_samples):

            beta_tilde[idx] = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde



    def getPosterior(self, X, Y, beta_rsi, recv_time):#rng, n, X, Y, EMitr=1):

        ######### posterior computation RSI #########
        pi_0 = np.ones((self.n,1)) * 1. / self.n
        beta_hat_rsi = np.zeros((self.n,1))
#        print('X shape: ',X.shape)

        beta_rsi = copy.deepcopy(self.beta)

        for i in range(X.shape[0]):
            for j in range(self.n):
                b = np.zeros((self.n,1))
                b[j,:] = self.mu
                # print('X[i] shape: ', X[i].shape)
                #assert X[i].shape == (1,32)
                # print('shape: ', Y[i].shape)
                pi_0[j] = np.float32(pi_0[j] * ss.norm(0,np.sqrt(self.sigma2)).pdf(Y[i] - np.dot(X[i], b)))
            pi_0 /= np.sum(pi_0)

        # return pi_0, pi_0, beta_rsi, None, None
            if np.amax(pi_0) == 0.:
                print('process ',os.getpid(),' would raise ValueError')
                break
                # raise ValueError('pi_0 max value 0.!')
            maxidxs = np.argwhere(pi_0 == np.amax(pi_0))
            eps = 0.
            for ids in maxidxs:
                eps += 1 - pi_0[ids][0]
            if(eps<self.err):
                # idxs = []
                for m in maxidxs:
                    beta_hat_rsi[m[0],:] = self.mu

                    beta_rsi[m[0],:] = 0.

                    # idxs.append(m[0])

                    # pi_0[m[0]] = 0.
                detected = np.count_nonzero(beta_hat_rsi)
#                print("so far detected: ",detected," left:",np.count_nonzero(beta_rsi))
                if np.count_nonzero(beta_rsi) == 0:
                    break
                #pi_0 /= np.sum(pi_0)

                idxs = [ii for ii in range(self.n) if ii not in np.nonzero(beta_hat_rsi)[0]]
                for idx in idxs:
                    b = copy.deepcopy(beta_hat_rsi)
                    b[idx,:] = self.mu
                    pi_0[idx] = 1.#1./self.n
                    for t in range(i+1):
                        pi_0[idx,:] = np.float32(pi_0[idx,:] * ss.norm(0,np.sqrt(self.sigma2)).pdf(Y[t] - np.dot(X[t],b)))

                #pi_0 = np.full((self.n,1), 1./(self.n - detected))
                pi_0[np.nonzero(beta_hat_rsi)[0],:] = 0.
                # pi_0[idxs] = 0.
                if np.amax(pi_0) == 0:
                    print('process ',os.getpid(),' would raise ValueError')
                    break
                    # raise ValueError('pi_0 max value 0.!')
#                print('sum:',np.sum(pi_0))

                pi_0 /= np.sum(pi_0)

                #print('sum: ',np.sum(pi_0))

                #beta_rsi = beta_rsi - beta_hat_rsi


        return beta_hat_rsi, pi_0, beta_rsi,None,None

    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        # rng = np.random.RandomState(trl)
        # n = beta.shape[0]

        wid =  qinfo.worker_id
        recv_time = qinfo.send_time
        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
            # print('worker #',wid,'writes up to sensing step',X.shape[0])
            # print('worker #',wid,'reads  up to sensing step',X.shape[0])
            # Sig0 = points_dict['Sig0']
            # gamma = points_dict['gamma']
            # B = points_dict['B']
            beta_rsi = points_dict['par']
            beta_hat_rsi, pi_0,beta_rsi,_,_ = self.getPosterior(X,Y,beta_rsi,qinfo.send_time)
            print('RSI\\1. trial: ',self.trl,' worker # ',wid,' recv_time ',recv_time,' finished posterior compute')
        else:#this branch True on initial evaluations for agents with samples from prior
            beta_rsi = self.beta
            pi_0 = np.ones((self.n,1)) * 1. / self.n
            #x = np.ones((self.n,1))/np.sqrt(self.n)
            #print(np.transpose(x))
            # XT_X = np.matmul(x,np.transpose(x)) # X^T * X
            #epsilon = self.rng.randn()*np.sqrt(self.sigma2)
            #y = np.matmul(np.transpose(x),self.beta)+epsilon
            # XT_Y = np.matmul(x,y) # X^T * Y
            #X = np.transpose(x)
            #Y = y
            beta_hat_rsi = np.zeros((self.n,1))

        k = np.count_nonzero(self.beta)


        max_reward = -math.inf
        bestx = np.zeros((self.n,1))

        # tt = len(Y)
        # print('X shape: ',X.shape)

        # for i in range(0,self.n):
        #     for l in range(2,self.n-i):
        # for l in range(1,self.n):
        #     for i in range(0,self.n):
        for i in range(0,self.n):
            for l in range(1,self.n-i):
                x = np.zeros((self.n,1))
                x[i:i+l] = 1/np.sqrt(l)

                # RSI information Gain:
                p_0 = np.sum(pi_0[0:i]) + np.sum(pi_0[i+l:self.n])
                p_1 = np.sum(pi_0[i:i+l])
                lamda = self.mu * 1./np.sqrt(l)
                IG = 0
                IG += -(p_0 * np.log(p_0 * ss.norm(0,1).pdf(0) + p_1 * ss.norm(0,1).pdf(-lamda/np.sqrt(self.sigma2))))
                IG += -(p_1 * np.log(p_0 * ss.norm(0,1).pdf(lamda/np.sqrt(self.sigma2)) + p_1 * ss.norm(0,1).pdf(0)))

                #IG_RSI = IG#RSI_IG(np.array([p_0,p_1]),mu * 1./np.sqrt(l))
                reward = IG

                if(reward>max_reward):
                    max_reward = reward
                    bestx = x
    #        print('sensing matrix:',np.transpose(bestx))
        #%% take a new observation

        self.rng = np.random.RandomState(self.trl + recv_time)
        epsilon = self.rng.randn()*np.sqrt(self.sigma2)
        # print("RSI\\2. trial: ",self.trl," worker: ",wid,"recv_time: ",recv_time," epsilon: ",epsilon)


        y = np.matmul(np.transpose(bestx),beta_rsi)+epsilon
        # X = np.append(X,np.transpose(bestx),axis=0)
        # Y = np.append(Y,y,axis=0)

        if not qinfo.compute_posterior:
            #X = [np.transpose(X), bestx]#np.append(X, np.transpose(bestx),axis=0)
            #print('len X: ',len(X))
            #Y = [Y, y]#np.append(Y, y, axis=0)
            result = {'x':[bestx], 'y':[y], 'par':beta_rsi, 'pre-eval':True}
        else:
            result = {'x':bestx,'y':y,'par':beta_rsi}

        # print('RSI\\3. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished compute')
        with open(qinfo.result_file, 'wb') as f:
            pkl.dump(result, f)


        print('RSI\\4. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished writing')
        # print('pi_0: ',pi_0)
        # print('beta_hat_rsi: ',beta_hat_rsi)
        # print('agent ', wid, ' returned: ',result)

        return result
