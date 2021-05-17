'''
Code for the following paper:

Ramina Ghods, Arundhati Banerjee, Jeff Schneider, ``Decentralized Multi-Agent Active Search for Sparse Signals",
2021 Conference On Uncertainty in Artificial Intelligence (UAI)

(c) Ramina Ghods(rghods@cs.cmu.edu), Arundhati Banerjee(arundhat@andrew.cmu.edu)

In this file, we are coding the class for LATSI in algorithm 3
'''

import numpy as np
import math
import pickle as pkl
import scipy.stats as ss
import copy
from scipy.stats import invgauss
import scipy

class LATSI(object):

    def __init__(self, p, q, beta, mu, theta2, sigma2, lmbd, EMitr, err, n_agents, alpha, trl):
        self.beta = beta #true_beta
        self.mu = mu
        self.theta2 = theta2
        self.sigma2 = sigma2
        self.lmbd = lmbd
        self.EMitr = EMitr
        self.trl = trl
        self.n_agents = n_agents
        self.gamma = lmbd**2

        self.p = p
        self.q = q
        self.n = beta.shape[0]
        self.L = int(self.n)
        self.M = int(self.n / self.L)

        self.rng = np.random.RandomState(trl)
        self.trl = trl
        self.err = err

        self.alpha = alpha
        # self.gamma = self.rng.rand(self.M)
        # self.B = self.rng.rand(self.L,self.L)
        # self.Sig0 = np.kron(np.diag(self.gamma),self.B)
#
#        print('init TS')



    def sample_from_prior(self, num_samples):

        self.rng = np.random.RandomState(self.trl)
        beta_tilde = np.zeros((num_samples, self.n, 1))

        for idx in range(num_samples):

            beta_tilde[idx] = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde

    def sample_from_prior_per_worker(self, recv_time):

        self.rng = np.random.RandomState(self.trl+recv_time)

        beta_tilde = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde



    def getPosterior(self, X, Y, L, recv_time):#rng, n, X, Y, EMitr=1):

        self.rng = np.random.RandomState(self.trl+recv_time)
        XT_X = np.matmul(np.transpose(X),X)
        XT_Y = np.matmul(np.transpose(X),np.reshape(Y[:,0],(-1,1)))
        U = np.diag(np.ones(self.n)*1)
        for j in range(self.EMitr): # we use EM as estimator for now
            tinv = np.linalg.inv(self.sigma2*np.eye(self.n)+np.matmul(np.matmul(U,XT_X),U))
            beta_hat = np.matmul(np.matmul(np.matmul(U,tinv),U),XT_Y)
            Sig_beta = np.matmul(np.matmul(U,tinv),U)*self.sigma2
            U = np.diag(np.squeeze(np.sqrt(np.absolute(beta_hat)/self.gamma)))

#        #sample with Gaussian prior:
#        beta_tilde = np.maximum(np.reshape(self.rng.multivariate_normal(np.squeeze(beta_hat),Sig_beta),(self.n,1)),np.zeros((self.n,1)))

        #sample with Laplace prior:
        beta_tilde = np.maximum(self.gibbs_invgauss(1000+recv_time,XT_X,XT_Y),np.zeros((self.n,1)))


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
                pi_0[j] = np.float32(pi_0[j] * ss.norm(0,np.sqrt(self.sigma2)).pdf(Y[i,1] - np.dot(X[i], b)))
            pi_0 /= np.sum(pi_0)

        # return pi_0, pi_0, beta_rsi, None, None
            if np.amax(pi_0) == 0.:
                break#raise ValueError('pi_0 max value 0.!')
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
                        pi_0[idx,:] = np.float32(pi_0[idx,:] * ss.norm(0,np.sqrt(self.sigma2)).pdf(Y[t,1] - np.dot(X[t],b)))

                #pi_0 = np.full((self.n,1), 1./(self.n - detected))
                pi_0[np.nonzero(beta_hat_rsi)[0],:] = 0.
                # pi_0[idxs] = 0.
                if np.amax(pi_0) == 0:
                    break
                print('sum:',np.sum(pi_0))

                pi_0 /= np.sum(pi_0)



        return beta_hat,U,beta_tilde,pi_0,beta_rsi

    def gibbs_invgauss(self,itr,XT_X,XT_Y):

        self.rng = np.random.RandomState(self.trl+itr-1000)
        tauinv_vec = 1/np.random.rand(self.n)
        for i in range(itr):
            Sig = np.linalg.inv(XT_X+self.sigma2*np.diag(tauinv_vec)+1e-3*np.eye(self.n))
            beta = self.get_multivariate_normal(np.squeeze(np.matmul(Sig,XT_Y)),self.sigma2*Sig,self.rng)
            for j in range(self.n):
                tauinv_vec[j] = invgauss.rvs(np.sqrt(self.sigma2)*(self.lmbd**(1/3))/np.abs(beta[j]))*(self.lmbd**(2/3))

        return np.reshape(beta,(self.n,1))

    def get_multivariate_normal(self,mean,cov,rng):
        # from np.dual import svd
        mean = np.array(mean)
        cov = np.array(cov)
        shape = []
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])
        x = rng.standard_normal(final_shape).reshape(-1, mean.shape[0])
        cov = cov.astype(np.double)
        # (u, s, v) = np.linalg.svd(cov, hermitian=True)

        (u, s, v) = scipy.linalg.svd(cov, lapack_driver='gesvd')

        x = np.dot(x, np.sqrt(s)[:, None] * v)
        x += mean
        x.shape = tuple(final_shape)

        return x



    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        recv_time = qinfo.send_time
        self.rng = np.random.RandomState(self.trl+recv_time)
        wid =  qinfo.worker_id
        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
#            print('worker #',wid,'writes up to sensing step',X.shape[0])
            print('worker #',wid,'reads  up to sensing step',X.shape[0])
            # Sig0 = points_dict['Sig0']
            # gamma = points_dict['gamma']
            # B = points_dict['B']
            L = points_dict['par']
            _,U,beta_tilde,pi_0,beta_rsi = self.getPosterior(X,Y,L,recv_time)
        else:#this branch True on initial evaluations for agents with samples from prior
            beta_tilde = self.sample_from_prior_per_worker(recv_time)
            beta_rsi = self.beta
            pi_0 = np.ones((self.n,1)) * 1. / self.n
            L = self.L


#        max_reward = -math.inf
        bestx = np.zeros((self.n,1))

        # print('X shape: ',X.shape)
        IG_RSI = np.zeros((self.q,self.q,self.p,self.p))-1000
        loss = np.zeros((self.q, self.q, self.p, self.p))+1000
        avg_loss = 0
        avg_IG_RSI = 0
        count = 0

        for i in range(self.q):
            for j in range(self.p):
                for del_x in range(1,self.q-i+1):
                    for del_y in range(1,self.p-j+1):
                        action = np.zeros((self.p, self.q))
                        action[j:(j+del_y),i:(i+del_x)] = 1./np.sqrt(del_x*del_y)
                        x = action.flatten().reshape((-1,1))
                        #print('i:',i,'j:',j,'del_x:',del_x,'del_y:',del_y)

        #for i in range(0,self.n):
        #    for l in range(1,self.n-i):
        #        x = np.zeros((self.n,1))
        #        x[i:i+l] = 1/np.sqrt(l)
                        if (qinfo.compute_posterior):
                            XT_X = np.matmul(np.transpose(X),X)
                            XT_Y = np.matmul(np.transpose(X),np.reshape(Y[:,0],(-1,1)))
                            tmp = np.linalg.inv(self.sigma2*np.eye(self.n)+np.matmul(np.matmul(U,XT_X+np.matmul(x,np.transpose(x))),U))
                            a = np.matmul(np.matmul(U,tmp),U)
                            xTb = np.ndarray.item(np.matmul(np.transpose(x),beta_tilde))
                            aXTY_b = np.matmul(a,XT_Y)-beta_tilde
                            ax = np.matmul(a,x)

                            loss[i,del_x-1,j,del_y-1] = (np.linalg.norm(aXTY_b)**2+
                                 (self.sigma2+xTb**2)*(np.linalg.norm(ax)**2)+
                                 np.ndarray.item(2*xTb*np.matmul(np.transpose(ax),aXTY_b)))
                        else:
                            U = np.eye(self.n)*np.sqrt(self.mu)
                            tmp = np.linalg.inv(self.sigma2*np.eye(self.n)+np.matmul(np.matmul(U,np.matmul(x,np.transpose(x))),U))
                            a = np.matmul(np.matmul(U,tmp),U)
                            xTb = np.ndarray.item(np.matmul(np.transpose(x),beta_tilde))
                            aXTY_b = -beta_tilde
                            ax = np.matmul(a,x)

                            loss[i,del_x-1,j,del_y-1] = (np.linalg.norm(aXTY_b)**2+(self.sigma2+xTb**2)*(np.linalg.norm(ax)**2)+np.ndarray.item(2*xTb*np.matmul(np.transpose(ax),aXTY_b)))

                        t = np.zeros((self.p, self.q))
                        t[j:(j+del_y),i:(i+del_x)] = 1
                        t = t.flatten().reshape((-1,1))
                        p_0 = np.sum(pi_0 * (1-t))
                        p_1 = np.sum(pi_0 * t)
                        # RSI information Gain:
                        #p_0 = np.sum(pi_0[0:i]) + np.sum(pi_0[i+l:self.n])
                        #p_1 = np.sum(pi_0[i:i+l])
                        lamda = self.mu * 1./np.sqrt(del_x*del_y) #np.sqrt(l)
                        IG = 0
                        if(np.sum(pi_0) != 0):
                            IG += -(p_0 * np.log(p_0 * ss.norm(0,1).pdf(0) + p_1 * ss.norm(0,1).pdf(-lamda/np.sqrt(self.sigma2))))
                            IG += -(p_1 * np.log(p_0 * ss.norm(0,1).pdf(lamda/np.sqrt(self.sigma2)) + p_1 * ss.norm(0,1).pdf(0)))
                        IG_RSI[i,del_x-1,j,del_y-1] = IG

                        avg_loss += loss[i,del_x-1,j,del_y-1]
                        avg_IG_RSI += IG_RSI[i,del_x-1,j,del_y-1]
                        count += 1

#                if(reward>max_reward):
#                    max_reward = reward
#                    bestx = x
#                    bestinv = b

        avg_loss /= count
        avg_IG_RSI /= count
        if(avg_IG_RSI==0):
            reward = - loss/avg_loss
        else:
            reward = - (self.alpha*(loss/avg_loss)) + IG_RSI/avg_IG_RSI
        [i,del_x,j,del_y] = np.unravel_index(reward.argmax(), reward.shape)
        bestx = np.zeros((self.p,self.q))#self.n,1))
        del_x += 1
        del_y += 1
        bestx[j:(j+del_y),i:(i+del_x)] = 1/np.sqrt(del_x * del_y)
        bestx = bestx.flatten().reshape((-1,1))
    #        print('sensing matrix:',np.transpose(bestx))
        #%% take a new observation
        epsilon = self.rng.randn()*np.sqrt(self.sigma2)
        print(epsilon)
        y = np.zeros((1,2))
        y[:,0] = np.matmul(np.transpose(bestx),self.beta)+epsilon
        y[:,1] = np.matmul(np.transpose(bestx),beta_rsi)+epsilon

        # X = np.append(X,np.transpose(bestx),axis=0)
        # Y = np.append(Y,y,axis=0)

        #update estimate, compute recovery
        # if qinfo.compute_posterior:
        #     X = np.append(X,np.transpose(bestx),axis=0)
        #     Y = np.append(Y,y,axis=0)
        # else:
        #     X = np.transpose(bestx)
        #     Y = y

        # beta_hat,_,_,_,_ = self.getPosterior(X,Y,L,recv_time)

        # est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
        # real = (self.beta>0)
        # partial_recovery_rate_latsi = np.sum(est==real)/(self.n)
        # correct_LATSI = 0.0
        # if(np.all(est==real)):
        #     correct_LATSI = 1.0


        if not qinfo.compute_posterior:
            result = {'x':[bestx], 'y':[y], 'par':L, 'pre-eval':True}#, 'full_recovery_rate':correct_LATSI, 'partial_recovery_rate':partial_recovery_rate_latsi}
        else:
            result = {'x':bestx,'y':y,'par':L}#, 'full_recovery_rate':correct_LATSI, 'partial_recovery_rate':partial_recovery_rate_latsi}

        with open(qinfo.result_file, 'wb') as f:
            pkl.dump(result, f)

        # with open('result.txt','a') as f:
        #     pkl.dump(result, f)

#        print('agent ', wid, ' returned: ',result)

        return result
