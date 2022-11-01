from utils import *
import numpy as np
import scipy.linalg as spla
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt

def upper_residual_spectrum(Res, spectrum, k):
    '''
    Input
    -----
    * Res: A - \wh{A}_l
    * spectrum: shape=(r,) spectrum of A
    Output
    ------
    * sin(\theta_i): shape=(k,), canonical angle in ascending order
    '''
    sigmaRes = spla.svd(Res, compute_uv=False, lapack_driver='gesvd')
    svra = np.minimum(sigmaRes[0]/spectrum[:k], np.flip(sigmaRes[:k])/spectrum[k-1])
    return svra

def estimate_random_embedding(spec, r, l, k, 
                            power_iter=0, algo=None, VTOmega=None,
                            embedding=lambda d,l: np.random.randn(d,l)/np.sqrt(l),
                            repeat=1, return_all_trails=True):
    '''
    * Omgwt: shape=(r,l), (Omgwt= or E[Omgwt]=) V.T@Omega  
    * spec: shape=(r,), descending order, non-negative
    * k: int, k<l
    * algo: RSVD with power iterations by default; 'bk' for block Krylov
    '''
    repeat = max(repeat,1)
    sin_est_all = []
    for trail in range(repeat):
        r = min(r, spec.shape[0])
        if algo is 'bk':
            l = l-(l%(power_iter+1))
            b = l//(power_iter+1)
            # polynomials for power-iteration
            # chebpoly = lambda x,l,u,q: chebval((l+u-2*x)/(u-l), 
            #                                     np.array([0]*(2*q+1)+[1]))/chebval((l+u)/(u-l), 
            #                                     np.array([0]*(2*q+1)+[1]))
            # pstar = lambda x,q: chebpoly(x, spec[-1], spec[0], q)
            # pstar = lambda x,q: chebpoly(x, 0.0, spec[0], q)
            pstar = lambda x,q: x**(2*q+1)
            psigma = np.concatenate([np.tile(pstar(spec,q),(b,1)) for q in range(power_iter+1)]).T #(r,l)
            Omgwt = np.tile(embedding(r,b),(1,power_iter+1)) #(r,l)
            Omega1 = Omgwt[:k] #(k,l)
            Omega2 = Omgwt[k:] #(r-k,l)
            Omgaux = matdivide(psigma[k:]*Omega2, psigma[:k]*Omega1) #(r-k,k)
            alpha = spla.svd(Omgaux, compute_uv=False) #(min(r-k,k),)
        else:
            if VTOmega is None or VTOmega.shape[0]!=r or VTOmega.shape[1]<k:
                Omgwt = embedding(r,l)
            else:
                Omgwt = VTOmega
            Omega1 = Omgwt[:k] #(k,l)
            Omega2 = Omgwt[k:] #(r-k,l)
            # polynomial for power-iteration
            pstar = lambda x: x**(2*power_iter+1)      
            Omgaux = matdivide(pstar(spec[k:]).reshape((-1,1))*Omega2, Omega1) #(r-k,k)
            alpha = spla.svd(Omgaux/pstar(spec[:k]).reshape((1,-1)), compute_uv=False) #(min(r-k,k),)  
        sin_est_all.append(np.flip(alpha / np.sqrt(1+alpha**2)))
    sin_est_all = np.array(sin_est_all)
    if return_all_trails:
        return sin_est_all.mean(axis=0), sin_est_all
    else:
        return sin_est_all.mean(axis=0)



def test_svra_computable(target, k, l=None,
                         algo='rsvd',
                         power_iter=0,
                         verbose=False):
    '''
    Input
    -----
    * target: dict('A':ndarray(shape=(m,n)), 
                   'U':ndarray(shape=(m,r)), 
                   's':ndarray(shape=(r,)), 
                   'V':ndarray(shape=(n,r)),
                   'tag':string)
    '''
    # [lower, upper]
    sin_left = {'truel': np.zeros(k),
                    'truek': np.zeros(k),
                    'res_sptrue': np.zeros((2,k)),
                    'res_spest': np.zeros((2,k))}
    sin_right = {'truel': np.zeros(k),
                     'truek': np.zeros(k),
                     'res_sptrue': np.zeros((2,k)),
                     'res_spest': np.zeros((2,k))}
    A = target['A']
    if l is None:
        l = k + max(k//2,10)
    # rsvd
    if algo is 'bk':
        Uap, sap, Vhap, Omega = rsvd(A,k,l=l//(power_iter+1), power_iter=power_iter, algo=algo, return_Omega=True)
    else:
        Uap, sap, Vhap, Omega = rsvd(A,k,l=l, power_iter=power_iter, algo=algo, return_Omega=True)
    # true canonical angles: (k,)
    sin_left['truel'],_ = canonical_angles(target['U'][:,:k],Uap,check_ortho=True)
    sin_left['truek'],_ = canonical_angles(target['U'][:,:k],Uap[:,:k],check_ortho=True)
    sin_right['truel'],_ = canonical_angles(target['V'][:,:k],Vhap.T,check_ortho=True)
    sin_right['truek'],_ = canonical_angles(target['V'][:,:k],Vhap[:k,:].T,check_ortho=True)
    # residual + true spectrum
    sigmaE = spla.svd(A - (Uap*sap.reshape((1,-1)))@Vhap, compute_uv=False, lapack_driver='gesvd')
    sin_left['res_sptrue'][1] = sigmaE[0]/target['s'][:k]
    # residual + estimated spectrum
    sin_left['res_spest'][1] = sigmaE[0]/sap[:k]
    output = {'target_tag': target['tag'],
              'k': k, 'l':l,
              'power_iter': power_iter,
              'sin_left': sin_left,
              'sin_right': sin_right}
    if verbose:
        plt.plot(sin_left['truel'], 'k-', label='$\sin(U_k,X)$')
        plt.plot(sin_left['truek'], 'k--', label='$\sin(U_k,\widehat U_k)$')
        plt.plot(sin_left['res_sptrue'][1], 'r-', label='$||A - \widehat A_l||_2 / \sigma_{[k]}$')
        plt.plot(sin_left['res_spest'][1], 'r--', label='$||A - \widehat A_l||_2 / \widehat\sigma_{[k]}$')
        plt.plot(np.flip(sigmaE[:k])/target['s'][k-1], 'b-', label='$\sigma_{[k]}(A - \widehat A_l) / \sigma_k$')
        plt.plot(np.flip(sigmaE[:k])/sap[k-1], 'b--', label='$\sigma_{[k]}(A - \widehat A_l) / \widehat\sigma_k$')
        plt.legend(fontsize=12)
        plt.xlabel('i')
        plt.hlines(1.0,0,k,linestyles='dotted')
        plt.hlines(0.1,0,k,linestyles='dotted')  
    return output


def test_svra(target, k, l=None,
              algo='rsvd', 
              power_iter=0,
              verbose=False,
              padding=False,
              repeat=5):
    '''
    Input
    -----
    * target: dict('A':ndarray(shape=(m,n)), 
                   'U':ndarray(shape=(m,r)), 
                   's':ndarray(shape=(r,)), 
                   'V':ndarray(shape=(n,r)),
                   'tag':string)
    '''
    # [lower, upper]
    sin_left = {'truel': np.zeros(k),
                'truek': np.zeros(k),
                'res_sptrue': np.zeros((2,k)),
                'res_spest': np.zeros((2,k)),
                'alpha_true': np.zeros(k),
                'alpha_est':np.zeros(k),
                'alpha_approxVh':np.zeros(k),
                'alpha_approxVh_est':np.zeros(k)}
    sin_right = {'truel': np.zeros(k),
                'truek': np.zeros(k),
                'res_sptrue': np.zeros((2,k)),
                'res_spest': np.zeros((2,k)),
                'alpha_true': np.zeros(k),
                'alpha_est':np.zeros(k),
                'alpha_approxVh':np.zeros(k),
                'alpha_approxVh_est':np.zeros(k)}
    A = target['A']
    if l is None:
        l = k + max(k//2,10)
    # rsvd
    if algo is 'bk':
        Uap, sap, Vhap, Omega = rsvd(A,k,l=l//(power_iter+1), power_iter=power_iter, algo=algo, return_Omega=True)
    else:
        Uap, sap, Vhap, Omega = rsvd(A,k,l=l, power_iter=power_iter, algo=algo, return_Omega=True)
    if padding and (len(target['s'])>len(sap)):
        sap_padded = np.concatenate((sap, np.ones(len(target['s'])-len(sap))*sap[-1]))
    else:
        sap_padded = np.array(sap)
    
    # true canonical angles: (k,)
    sin_left['truel'],_ = canonical_angles(target['U'][:,:k],Uap,check_ortho=True)
    sin_left['truek'],_ = canonical_angles(target['U'][:,:k],Uap[:,:k],check_ortho=True)
    sin_right['truel'],_ = canonical_angles(target['V'][:,:k],Vhap.T,check_ortho=True)
    sin_right['truek'],_ = canonical_angles(target['V'][:,:k],Vhap[:k,:].T,check_ortho=True)
    
    # residual + true spectrum
    Res = A - (Uap*sap.reshape((1,-1)))@Vhap
    sin_left['res_sptrue'][1] = upper_residual_spectrum(Res, target['s'], k)
    # residual + estimated spectrum
    sin_left['res_spest'][1] = upper_residual_spectrum(Res, sap_padded, k)
    
    # true alpha's
    r = target['s'].shape[0]
    sin_left['alpha_true'], left_true_all = estimate_random_embedding(target['s'], r, l, k, 
                                                                      power_iter=power_iter, algo=algo, repeat=repeat)
    # estimated alpha's
    sin_left['alpha_est'], left_est_all = estimate_random_embedding(sap_padded, r, l, k, 
                                                                    power_iter=power_iter, algo=algo, repeat=repeat)
    # estimated alpha with approx V
    VTOmega = Vhap @ Omega
    sin_left['alpha_approxVh'], left_approxVh_all = estimate_random_embedding(target['s'], r, l, k, 
                                                                    power_iter=power_iter, algo=algo, VTOmega=VTOmega, repeat=repeat)
    sin_left['alpha_approxVh_est'], left_approxVh_est_all = estimate_random_embedding(sap_padded, r, l, k, 
                                                                    power_iter=power_iter, algo=algo, VTOmega=VTOmega, repeat=repeat)                                                             
    # assemble output
    output = {'target_tag': target['tag'],
              'k': k, 'l':l,
              'sin_left': sin_left,
              'sin_right': sin_right}
    if verbose:
        plt.plot(sin_left['truel'], 'k-', label='$\sin(U_k,X)$')
        plt.plot(sin_left['truek'], 'k--', label='$\sin(U_k,\widehat U_k)$')
        plt.plot(sin_left['res_sptrue'][1], 'b-', label='residual + true $\sigma(A)$')
        plt.plot(sin_left['res_spest'][1], 'b--', label='residual + approx $\sigma(A)$')
        xx = np.arange(sin_left['alpha_true'].shape[0])
        plt.plot(sin_left['alpha_true'], 'r-', label='true $\sigma(A)$ + independent $\Omega$')
        plt.fill_between(xx, left_true_all.min(axis=0), left_true_all.max(axis=0), color='r', alpha=0.2)
        plt.plot(sin_left['alpha_est'], 'r--', label='estimated $\sigma(A)$ + independent $\Omega$')
        plt.fill_between(xx, left_est_all.min(axis=0), left_est_all.max(axis=0), color='r', alpha=0.2)
        plt.plot(sin_left['alpha_approxVh'], 'g-', label='true $\sigma(A)$ + $\widehat{V}^*\Omega$')
        # plt.fill_between(xx, left_approxVh_all.min(axis=0), left_approxVh_all.max(axis=0), color='g', alpha=0.2)
        plt.plot(sin_left['alpha_approxVh_est'], 'g--', label='estimated $\sigma(A)$ + $\widehat{V}^*\Omega$')
        # plt.fill_between(xx, left_approxVh_est_all.min(axis=0), left_approxVh_all.max(axis=0), color='g', alpha=0.2)
        # plt.legend(fontsize=12)
        plt.xlabel('i')
        plt.hlines(1.0,0,k,linestyles='dotted')
        plt.hlines(0.1,0,k,linestyles='dotted')  
    return output