#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.spatial.distance import cdist
from numpy import random,matlib,linalg
from PIL import Image
class pycolor:
    RED = '\033[31m'
    END = '\033[0m'


# # Dataset


Visual_category = loadmat('../data/visual_category.mat')
v_candidate = Visual_category['VGG19_candidate'].T
c_candidate = Visual_category['word2vec_candidate'].T
def dataset(subject):
    fMRI = loadmat('../data/fMRI_data/subject0{}.mat'.format(subject))
    f_train = fMRI['sub0{}_train'.format(subject)].T
    f_test = fMRI['sub0{}_test_ave'.format(subject)].T
    if subject == 1 or subject == 2 or subject == 3:
        v_train = Visual_category['VGG19_train'].T
        c_train = Visual_category['word2vec_train'].T
    else:
        v_train = Visual_category['VGG19_train_sub0{}'.format(subject)].T
        c_train = Visual_category['word2vec_train_sub0{}'.format(subject)].T
    print('Voxels : {}'.format(f_train.shape[0]))
    return f_train,v_train,c_train,f_test


# # Parameters


def parameter(f_train,v_train,c_train,f_test):
    N = f_train.shape[1]
    N_test = f_test.shape[1]
    D = [f_train.shape[0],v_train.shape[0],c_train.shape[0]]
    Dz = min(D[0],D[1],D[2])
    print('Dimensions of latent variables : {}'.format(Dz))
    return N,N_test,D,Dz


# # Normalize


def normalize(f_train,v_train,c_train):
    N = np.size(f_train,1)
    X_mean = [np.mean(f_train,axis=1),np.mean(v_train,axis=1),np.mean(c_train,axis=1)]
    X_norm = [np.std(f_train,axis=1,ddof=1),np.std(v_train,axis=1,ddof=1),np.std(c_train,axis=1,ddof=1)]
    X_train = [f_train-matlib.repmat(X_mean[0],N,1).T,v_train-matlib.repmat(X_mean[1],N,1).T,c_train-matlib.repmat(X_mean[2],N,1).T]
    X = [X_train[0]/matlib.repmat(X_norm[0],N,1).T,X_train[1]/matlib.repmat(X_norm[1],N,1).T,X_train[2]/matlib.repmat(X_norm[2],N,1).T]
    return X,X_mean,X_norm
def normalize_item(item,X_mean,X_norm):
    N_item = np.size(item,1)
    item = item-matlib.repmat(X_mean,N_item,1).T
    item = item/matlib.repmat(X_norm,N_item,1).T
    return item
def renormalize_item(item,X_mean,X_norm):
    N_item = np.size(item,1)
    item = item*matlib.repmat(X_norm,N_item,1).T
    item = item+matlib.repmat(X_mean,N_item,1).T
    return item


# # Hyper-parameters


subject = 3 # subject index
maxiter = 10 # number of updating model parameters
thres_a_inv = 1e-1 # ARD parameter
eta = 0.5 # trade-off parameter between visual features and category features
N_trial = 100 # trial numbers of model training and prediction (N_trial was set to 1000 in the original paper.)


# # Initialize


def initialize(X,N,D,Dz):   
    # Z
    Z = random.randn(Dz,N)
    SigmaZ_inv = np.eye(Dz)
    SZZ = Z@Z.T + N*SigmaZ_inv
    SZZrep = [matlib.repmat(np.diag(SZZ),D[0],1),matlib.repmat(np.diag(SZZ),D[1],1),matlib.repmat(np.diag(SZZ),D[2],1)]
    # alpha,gamma
    A_inv = [np.ones((D[0],Dz)),np.ones((D[1],Dz)),np.ones((D[2],Dz))]
    A0_inv = [np.zeros((D[0],Dz)),np.zeros((D[1],Dz)),np.zeros((D[2],Dz))]
    gamma0 = [np.zeros((D[0],Dz)),np.zeros((D[1],Dz)),np.zeros((D[2],Dz))]
    gamma = [1/2+gamma0[0],1/2+gamma0[1],1/2+gamma0[2]]
    gamma_xx = [np.sum(X[0]**2)/2,np.sum(X[1]**2)/2,np.sum(X[2]**2)/2]
    gamma_beta = [D[0]*N/2,D[1]*N/2,D[2]*N/2]
    # beta
    beta_inv = [1,1,1]
    return Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv


# # Update


def update(X,N,Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv,D):
    # initialize
    SigmaW_inv = [0]*3
    W = [0]*3
    WW = [0]*3
    beta_inv_gamma = [0]*3
    print ('********************subject={},trial={},iteration={}'.format(subject,t,maxiter))
    for l in range(maxiter):
        # W-step
        for i in range(3):
            SigmaW_inv[i] = A_inv[i]/((1/beta_inv[i])*SZZrep[i]*A_inv[i]+1)
            W[i] = (1/beta_inv[i])*X[i]@Z.T*SigmaW_inv[i]
            WW[i] = np.diag(SigmaW_inv[i].sum(axis=0))+W[i].T@W[i]
        # Z-step
        SigmaZ = (1/beta_inv[0])*WW[0]+(1/beta_inv[1])*WW[1]+(1/beta_inv[2])*WW[2]+np.eye(Dz)
        SigmaZ_inv = linalg.inv(SigmaZ)
        Z =  (1/beta_inv[0])*SigmaZ_inv@W[0].T@X[0]+(1/beta_inv[1])*SigmaZ_inv@W[1].T@X[1]+(1/beta_inv[2])*SigmaZ_inv@W[2].T@X[2]
        SZZ = Z@Z.T + N*SigmaZ_inv
        SZZrep = [matlib.repmat(np.diag(SZZ),D[0],1),matlib.repmat(np.diag(SZZ),D[1],1),matlib.repmat(np.diag(SZZ),D[2],1)]
        for i in range(3):
            # alpha-step
            A_inv[i] = (W[i]**2/2+SigmaW_inv[i]/2+gamma0[i]*A0_inv[i])/gamma[i]
            # beta-step
            beta_inv_gamma[i] = gamma_xx[i]-np.trace(W[i]@Z@X[i].T)+np.trace(SZZ@WW[i])/2
            beta_inv[i] = beta_inv_gamma[i]/gamma_beta[i]
        # find irrelevance parameters
        a_inv = [A_inv[0].sum(axis=0),A_inv[1].sum(axis=0),A_inv[2].sum(axis=0)]
        a_inv_max = [max(a_inv[0]),max(a_inv[1]),max(a_inv[2])]
        ix_a = [a_inv[0]>a_inv_max[0]*thres_a_inv, a_inv[1]>a_inv_max[1]*thres_a_inv, a_inv[2]>a_inv_max[2]*thres_a_inv]
        ix_z = np.logical_and(ix_a[0],ix_a[1],ix_a[2])
    print('Effect number of dimensions (ARD) : {}'.format(np.sum(ix_z)))
    return W,WW,beta_inv,Z,X


# # Predict


def predict(W,WW,beta_inv,f_test,D,Dz):
    # calculate posterior z from fMRI activity
    SigmaZnew = (1/beta_inv[0])*WW[0]+np.eye(Dz)
    SigmaZnew_inv = linalg.inv(SigmaZnew)
    prZ = SigmaZnew_inv@((1/beta_inv[0])*W[0].T@f_test)
    # predictive distribution
    v_pred = W[1]@prZ
    v_pred_cov = W[1]@SigmaZnew_inv@W[1].T+beta_inv[1]*np.eye(D[1])
    c_pred = W[2]@prZ
    c_pred_cov = W[2]@SigmaZnew_inv@W[2].T+beta_inv[2]*np.eye(D[2])
    return v_pred,c_pred


# # Estimate image categories


def evaluate(V_pred,C_pred):
    # Estimate image categories from visual features
    v_corr = (1 - cdist(V_pred.T, v_candidate.T, metric='correlation'))
    # Estimate image categories from category features
    c_corr = (1 - cdist(C_pred.T, c_candidate.T, metric='correlation'))
    # Rankings of estimated image categories
    def calc_rank(corr):
        sort = np.sort(corr,axis=1)[:,::-1]
        sort_ix = np.argsort(corr,axis=1)[:,::-1]
        Rank = []
        for i in range(N_test):
            Rank.append(int(np.where(sort_ix[i,:]==i)[0]+1))
        return Rank,sort,sort_ix
    def calc_acc(corr):
        accuracy = []
        for i in range(np.size(corr,0)):
            correct = 0
            for j in range(np.size(corr,1)):
                if corr[i,i] > corr[i,j]:
                    correct += 1
            accuracy.append(correct/(np.size(corr,1)-1))
        return accuracy
    # fusion of estimated rankings
    corr_fusion = eta*v_corr+(1-eta)*c_corr
    Rank_fusion,candidate_corr,candidate_ix = calc_rank(corr_fusion)
    Acc_fusion = calc_acc(corr_fusion)
    test_Rank_fusion = np.mean(Rank_fusion)
    test_Acc_fusion = np.mean(Acc_fusion)
    print('Average ranks from fusion results : {}'.format(test_Rank_fusion))
    print('Average accuracy from fusion results : {}'.format(test_Acc_fusion))
    return Rank_fusion,candidate_ix,test_Rank_fusion,test_Acc_fusion


# # Generate N-trial samples


V_pred = C_pred = 0
f_train,v_train,c_train,f_test = dataset(subject)
N,N_test,D,Dz = parameter(f_train,v_train,c_train,f_test)
# normalize
X,X_mean,X_norm = normalize(f_train,v_train,c_train)
for t in range(N_trial):
    Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv = initialize(X,N,D,Dz)
    W,WW,beta_inv,Z,X_update = update(X,N,Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv,D)
    X_test = normalize_item(f_test,X_mean[0],X_norm[0])

    v_pred,c_pred = predict(W,WW,beta_inv,X_test,D,Dz)
    v_pred = renormalize_item(v_pred,X_mean[1],X_norm[1])
    c_pred = renormalize_item(c_pred,X_mean[2],X_norm[2])

    V_pred += v_pred
    C_pred += c_pred
    
# average of N-trials
V_pred_mean = V_pred/N_trial
C_pred_mean = C_pred/N_trial
print('****************************************Estimation Result')
Rank_fusion,candidate_ix,mean_rank,mean_acc = evaluate(V_pred_mean,C_pred_mean)


# # Estimated image categories for each test image


for test_index in range(1,51):
    # read image
    im = Image.open('../data/test_images/test{}.JPEG'.format(test_index))
    plt.imshow(im)
    plt.show()
    # read canididate
    f = open('../data/candidate_name.txt')
    candidate = f.readlines()
    print('test image category : {}'.format(candidate[test_index-1]))
    flag = 0
    for i in range(5):
        if i ==  Rank_fusion[test_index-1]-1:
            print(pycolor.RED + 'Rank {} : {}'.format(i+1,candidate[candidate_ix[test_index-1,i]]) + pycolor.END)
            flag = 1
        else:
            print('Rank {} : {}'.format(i+1,candidate[candidate_ix[test_index-1,i]]))
    if flag == 0:
        print('   *\n   *\n   *')
        print(pycolor.RED + 'Rank {} : {}'.format(Rank_fusion[test_index-1],candidate[test_index-1]) + pycolor.END)

