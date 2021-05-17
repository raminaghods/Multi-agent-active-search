'''
Code for the following paper:

Ramina Ghods, Arundhati Banerjee, Jeff Schneider, ``Decentralized Multi-Agent Active Search for Sparse Signals",
2021 Conference On Uncertainty in Artificial Intelligence (UAI)

(c) Ramina Ghods(rghods@cs.cmu.edu), Arundhati Banerjee(arundhat@andrew.cmu.edu)

Code for SPATS in algorithm 2

'''

import numpy as np
import math
import pickle as pkl
from scipy.stats import invgauss

import scipy
import scipy.stats as ss

class SPATS(object):

    def __init__(self, p, q, beta, mu, theta2, sigma2, lmbd, EMitr, n_agents, trl):
        self.beta = beta
        self.mu = mu
        self.theta2 = theta2
        self.sigma2 = sigma2
        self.lmbd = lmbd
        self.EMitr = EMitr
        self.trl = trl
        self.n_agents = n_agents

        #2D grid of shape pXq
        self.p = p
        self.q = q
        self.n = beta.shape[0]
        self.L = int(self.n)
        self.M = int(self.n / self.L)

        self.rng = np.random.RandomState(trl)

        # self.gamma = self.rng.rand(self.M)
        # self.B = self.rng.rand(self.L,self.L)
        # self.Sig0 = np.kron(np.diag(self.gamma),self.B)
#
#        print('init TS')


    def sample_from_prior(self, num_samples):

        beta_tilde = np.zeros((num_samples, self.n, 1))

        for idx in range(num_samples):

            beta_tilde[idx] = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde

    def sample_from_prior_per_worker(self, recv_time):
    	# beta_tilde = np.zeros((n,1))

    	self.rng = np.random.RandomState(self.trl+recv_time)

    	beta_tilde = self.rng.laplace(scale=1/self.lmbd, size=(self.n,1))

    	return beta_tilde



    def getPosterior(self, X, Y, L, recv_time):#rng, n, X, Y, EMitr=1):

        # L = n
        # M = int(n/L)


        if(L>int(self.n/self.n_agents)):
            L = int(self.n/self.n_agents)
        elif(L>1):
            L = int(L/2)
        else:
            L = 1
        M = int(self.n/L)

        #print('L:',L)

        tt = len(Y)

        self.rng = np.random.RandomState(self.trl+recv_time)

        gamma = self.rng.rand(M)
#        print("gamma: ",gamma)
        B = np.eye((L))
        Sig0 = np.kron(np.diag(gamma),B)

        for j in range(self.EMitr): # we use EM as estimator for now
            tinv = np.linalg.inv(self.sigma2*np.eye(tt)+np.matmul(np.matmul(X,Sig0),np.transpose(X)))
            F = np.matmul(np.matmul(Sig0,np.transpose(X)),tinv)
            beta_hat = np.matmul(F,Y)
            Sig_beta = Sig0 - np.matmul(F,np.matmul(X,Sig0))
            Bt = np.zeros((L,L))
            for j in range(M):
                Sig_beta_j = Sig_beta[j*L:(j+1)*L,j*L:(j+1)*L]
                mu_beta_j = beta_hat[j*L:(j+1)*L]
                Sig_mu = Sig_beta_j+np.matmul(mu_beta_j,np.transpose(mu_beta_j))
                gamma[j] = np.trace(np.matmul(np.linalg.inv(B),Sig_mu))/L
                Bt += Sig_mu/gamma[j]
            B = Bt/M
            Sig0 = np.kron(np.diag(gamma),B)

#       making Sig_beta positive semi definite
        min_eig = np.amin(np.real(np.linalg.eigvals(Sig_beta)))
        if min_eig < 0:
            Sig_beta -= 10*min_eig * np.eye(*Sig_beta.shape)

#        # checks if Sig_beta is positive semi-definite:
#        M = np.matrix(Sig_beta)
#        print(np.all(np.linalg.eigvals(M+M.transpose()) > 0) and np.allclose(Sig_beta, np.transpose(Sig_beta), rtol=1e-4, atol=1e-4))


#        #sample with Gaussian prior:
        beta_tilde = np.maximum(np.reshape(self.get_multivariate_normal(np.squeeze(beta_hat),Sig_beta,self.rng),(self.n,1)),np.zeros((self.n,1)))

        # beta_tilde = np.maximum(np.reshape(self.get_multivariate_normal(np.squeeze(beta_hat),Sig_beta,self.rng),(self.n,1)),np.zeros((self.n,1)))


        #sample with Laplace prior:
        #XT_X = np.matmul(np.transpose(X),X)
        #XT_Y = np.matmul(np.transpose(X),Y)
        #beta_tilde = np.maximum(self.gibbs_invgauss(1000+recv_time,XT_X,XT_Y),np.zeros((self.n,1)))

        return beta_hat,Sig0,gamma,beta_tilde,L

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


    def gibbs_invgauss(self,itr,XT_X,XT_Y):
        self.rng = np.random.RandomState(self.trl+itr) #itr includes recv_time, so didn't pass/use that explicitly
        tauinv_vec =  1/self.rng.rand(self.n)#1/np.random.rand(self.n)
        for i in range(itr):
            Sig = np.linalg.inv(XT_X+self.sigma2*np.diag(tauinv_vec))
            beta = self.rng.multivariate_normal(np.squeeze(np.matmul(Sig,XT_Y)),self.sigma2*Sig)
            for j in range(self.n):
                tauinv_vec[j] = invgauss.rvs(np.sqrt(self.sigma2)*(self.lmbd**(1/3))/np.abs(beta[j]))*(self.lmbd**(2/3))

        return np.reshape(beta,(self.n,1))


    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        # rng = np.random.RandomState(trl)
        # n = beta.shape[0]

        recv_time = qinfo.send_time
        wid =  qinfo.worker_id
        #print('seed number',self.trl + recv_time)
        self.rng = np.random.RandomState(self.trl + recv_time)
        print('recv_time',recv_time)
        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
#            print('worker #',wid,'writes up to sensing step',X.shape[0])
            #print('worker #',wid,'reads  up to sensing step',X.shape[0])
            # Sig0 = points_dict['Sig0']
            # gamma = points_dict['gamma']
            # B = points_dict['B']
            L = points_dict['par']
            _,Sig0,gamma,beta_tilde,L = self.getPosterior(X,Y,L,recv_time)
        else:#this branch True on initial evaluations for agents with samples from prior
            beta_tilde = self.sample_from_prior_per_worker(recv_time) #points_dict['beta']
            L = self.L



        k = np.count_nonzero(self.beta)


        max_reward = -math.inf
        bestx = np.zeros((self.n,1))


        # print('X shape: ',X.shape)
        for i in range(self.q):
            for j in range(self.p):
                for del_x in range(1,self.q-i+1):
                    for del_y in range(1,self.p-j+1):
                        action = np.zeros((self.p, self.q))
                        action[j:(j+del_y),i:(i+del_x)] = 1./np.sqrt(del_x*del_y)
                        x = action.flatten().reshape((-1,1))

                # x = np.zeros((self.n,1))
                # x[i:i+l] = 1/np.sqrt(l)
                        if (qinfo.compute_posterior):
                            tt = len(Y)
                            Xt = np.append(X,np.transpose(x),axis=0)
                            tmp = np.linalg.inv(self.sigma2*np.eye(tt+1)+np.matmul(np.matmul(Xt,Sig0),np.transpose(Xt)))
                            b = np.matmul(np.matmul(Sig0,np.transpose(Xt)),tmp)
                            b1 = b[:,:-1]
                            # print("shape::: ",b1.shape) #128,2
                            b2 = np.reshape(b[:,-1],(-1,1))
                            # print("shape:::",b2.shape) 128,1
                            xTb = np.ndarray.item(np.matmul(np.transpose(x),beta_tilde))
                            # print("shape:::",xTb.shape)
                            b1Y_b = np.matmul(b1,Y)-beta_tilde
                            # print("Y shape::",Y.shape)
                            # print("shape:::",(2*xTb*np.matmul(np.transpose(b2),b1Y_b)).shape)
                            reward = -1*(np.linalg.norm(b1Y_b)**2+
                                         (self.sigma2+xTb**2)*(np.linalg.norm(b2)**2)+
                                         np.ndarray.item(2*xTb*np.matmul(np.transpose(b2),b1Y_b)))
                        else:
                            gamma = np.ones(self.n_agents)
                            B = np.ones((int(self.n/self.n_agents),int(self.n/self.n_agents)))*self.mu/2+np.eye(int(self.n/self.n_agents))*self.mu/2
                            Sig0 = np.kron(np.diag(gamma),B)
                            tmp = np.linalg.inv(self.sigma2+np.matmul(np.matmul(np.transpose(x),Sig0),x))
                            b = np.matmul(np.matmul(Sig0,x),tmp)
                            xTb = np.ndarray.item(np.matmul(np.transpose(x),beta_tilde))
                            reward = -1*(np.linalg.norm(beta_tilde)**2+
                                         (self.sigma2+xTb**2)*(np.linalg.norm(b)**2)+
                                         np.ndarray.item(2*xTb*np.matmul(np.transpose(b),beta_tilde)))
                        if(reward>max_reward):
                            max_reward = reward
                            bestx = x
                            bestinv = b
        #print('sensing matrix:',np.transpose(bestx))
        #%% take a new observation
        # recv_time = qinfo.send_time
        # self.rng = np.random.RandomState(recv_time)
        epsilon = self.rng.randn()*np.sqrt(self.sigma2)
        y = np.matmul(np.transpose(bestx),self.beta)+epsilon

        # if qinfo.compute_posterior:
        #     X = np.append(X,np.transpose(bestx),axis=0)
        #     Y = np.append(Y,y,axis=0)
        # else:
        #     X = np.transpose(bestx)
        #     Y = y

        # beta_hat,_,_,_,_ = self.getPosterior(X,Y,L,recv_time)

        # est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
        # real = (self.beta>0)
        # partial_recovery_rate_spats = np.sum(est==real)/(self.n)
        # correct_SPATS = 0.0
        # if(np.all(est==real)):
        #     correct_SPATS = 1.0

        if not qinfo.compute_posterior:
            #X = [np.transpose(X), bestx]#np.append(X, np.transpose(bestx),axis=0)
            #print('len X: ',len(X))
            #Y = [Y, y]#np.append(Y, y, axis=0)
            result = {'x':[bestx], 'y':[y], 'par':L, 'pre-eval':True}#, 'full_recovery_rate':correct_SPATS, 'partial_recovery_rate':partial_recovery_rate_spats}
        else:
            result = {'x':bestx,'y':y,'par':L}#, 'full_recovery_rate':correct_SPATS, 'partial_recovery_rate':partial_recovery_rate_spats}

        with open(qinfo.result_file, 'wb') as f:
            pkl.dump(result, f)

        # with open('result.txt','a') as f:
        #     pkl.dump(result, f)

#        print('agent ', wid, ' returned: ',result)

        return result
