# utils.py
import numpy as np
import scipy.linalg as spla
import tensorflow as tf

def power_iteration(A,Omega,p, krylov=False):
    '''
    A: ndarray(shape=(m,n))
    Omega: ndarray(shape=(n,l))
    p: non-negative int
    Acol: ndarray(shape=(m,l)), ((A@A.T)**p)@A@Omega with orthonormalization
    '''
    l = Omega.shape[-1]
    X = A@Omega #(m,l)
    Qcol,_ = spla.qr(X, mode='economic') #(m,l)
    for iter in range(p):
        if krylov:
            Qrow,_ = spla.qr(A.T@Qcol, mode='economic') #(n,l*(iter+1))
            Qcol,_ = spla.qr(A @ np.hstack((Qrow,Omega)),mode='economic') #(m,l*(iter+2))
        else:
            Qrow,_ = spla.qr(A.T@Qcol[:,-l:], mode='economic') #(n,l)
            Qcol,_ = spla.qr(A@Qrow,mode='economic') #(m,l)
    return Qcol


def rsvd(A, k, l=None, power_iter=0, algo='rsvd', return_Omega=False):
    '''
    * algo: 'rsvd'(default, Halko2011), 'gn'(generalized Nystrom), 'bk'(block Krylov)
    '''
    if l is None:
        l = min(k+10, A.shape[1])
    # construct low-rank approximation
    Omega = np.random.randn(A.shape[1],l)/np.sqrt(l) #(n,l)
    if algo is 'bk': # block krylov (ignoring l)
        Qcol = power_iteration(A,Omega,power_iter,krylov=True) #(m,l*(p+1))
        Uap_reduced, sap, Vhap = spla.svd(Qcol.T @ A, full_matrices=False, lapack_driver='gesvd') #(l*(p+1),l*(p+1)),(l*(p+1),),(l*(p+1),n)
        Uap = Qcol @ Uap_reduced #(m,l*(p+1))
    # elif algo is 'gn':
        # generalized Nystrom
    else: # rsvd
        Qcol = power_iteration(A,Omega,power_iter) #(m,l)
        Uap_reduced, sap, Vhap = spla.svd(Qcol.T @ A, full_matrices=False, lapack_driver='gesvd') #(l,l),(l,),(l,n)
        Uap = Qcol @ Uap_reduced #(m,l)
    # output
    if return_Omega:
        return Uap, sap, Vhap, Omega
    else:
        return Uap, sap, Vhap


def matdivide(A,B, compute_spectrum=False, tol=1e-9):
    '''
    A: (m,n) ndarray
    B: (k,n) ndarray
    compute A*pinv(B), or its spectrum
    '''
    U,s,Vh = spla.svd(B, full_matrices=False, lapack_driver='gesvd')
    r = np.count_nonzero(s>tol)
    Aux = (A @ Vh[:r].T)/s[:r].reshape(1,-1) #(m,r)
    if compute_spectrum:
        spectrum = spla.svd(Aux, compute_uv=False, lapack_driver='gesvd')
        return spectrum
    else:
        return (Aux @ U[:,:r].T) #(m,k)


def genenrate_mnist_target(size=int(1e3)):
    '''
    Output
    -----
    * target: dict('A':ndarray(shape=(m,n)), 
                   'U':ndarray(shape=(m,r)), 
                   's':ndarray(shape=(r,)), 
                   'V':ndarray(shape=(n,r)),
                   'tag':string)
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # assert x_train.shape == (60000, 28, 28)
    # assert x_test.shape == (10000, 28, 28)
    # assert y_train.shape == (60000,)
    # assert y_test.shape == (10000,)
    # ndarrays
    Xtrain = x_train[np.random.choice(x_train.shape[0]//10,size,replace=False)].reshape((size,-1))
    Xtrain = Xtrain / spla.norm(Xtrain, ord=2, axis=1, keepdims=True)
    U,s,Vh = spla.svd(Xtrain, full_matrices=False, lapack_driver='gesvd')
    target = {'A': Xtrain, 
              'U': U, 
              's': s,
              'V': Vh.T,
              'tag': 'mnist-train%d'%(size)}
    return target


def genenrate_gaussian_target(m=1000, n=1000, rank=1000, spectrum=None):
    '''
    Output
    -----
    * target: dict('A':ndarray(shape=(m,n)), 
                   'U':ndarray(shape=(m,r)), 
                   's':ndarray(shape=(r,)), 
                   'V':ndarray(shape=(n,r)),
                   'tag':string)
    '''
    if spectrum is None:
        spectrum = np.linspace(1.0, 1e-5, num=rank)
    elif len(spectrum)<rank:
        spectrum = np.concatenate((spectrum, np.zeros(rank-len(spectrum))))
    U,_ = spla.qr(np.random.randn(m,rank), mode='economic')
    V,_ = spla.qr(np.random.randn(n,rank), mode='economic')
    target = {'A': ( U*(spectrum[:rank].reshape((1,-1))) )@V.T, 
              'U': U, 
              's': spectrum[:rank],
              'V': V,
              'tag': 'Gaussian'}
    return target


def canonical_angles(U,V, check_ortho=False):
    '''
    Input
    -----
    * U: ndarray(shape=(d,k)) (d>k)
    * V: ndarray(shape=(d,l)) (d>l)
    Output
    ------
    * min(k,l) sin's and cos's, ascending
    '''
    if check_ortho:
        U,_ = spla.qr(U, mode='economic')
        V,_ = spla.qr(V, mode='economic')
    Mcos = U.T @ V # (k,l)
    cos = np.minimum(spla.svd(Mcos, compute_uv=False, lapack_driver='gesvd'),1.0)
    sin = np.sqrt(1-cos**2)
    return sin, cos