from scipy.io import loadmat, savemat
from scipy.linalg import block_diag
import numpy as np
from numpy import linalg as LA
#from PIL import Image
import datetime
from tqdm import tqdm

def main():
    # load dataset
    x3dl = loadmat('dataset/X3DL.mat')
    mask = loadmat('dataset/mask.mat')
    ottawa = loadmat('dataset/Ottawa.mat')
    X3D_corrupted = ottawa['X3D_ref'] * mask['mask_3D']
    print(X3D_corrupted.shape)
    X3d_rec = ADMM_ADAM(X3D_corrupted, mask['mask_3D'], x3dl['X3D_DL'])

    print(x3dl.keys())
    print(mask.keys())
    print(ottawa.keys())
    print("X3D_DL: ", x3dl['X3D_DL'].shape)
    print("mask_3D: ", mask['mask_3D'].shape)
    print("X3D_ref: ", ottawa['X3D_ref'].shape)

    # save mat
    savemat('dataset/save_X3DL.mat', x3dl)
    savemat('dataset/save_mask.mat', mask)
    savemat('dataset/save_Ottawa.mat', ottawa)
    savemat('dataset/X3D_rec.mat', {'X3D_rec':X3d_rec})


def ADMM_ADAM(X3D_corrupted, mask, x3dl):
    now = datetime.datetime.now()
    # para
    N=10 # dimension of the hyperspectral subspace
    lam=0.01 # regularization parameter
    mu=1e-3 # ADMM penalty parameter 

    # compute S_DL 
    row, col , all_bands = X3D_corrupted.shape
    spatial_len=row*col
    X2D_DL = x3dl.reshape(-1,all_bands).T
    # 計算講義E
    E = compute_basis(x3dl, N)
   # savemat('dataset/E.mat', {"E":E})

    S_DL = np.dot(E.T, X2D_DL)
    # savemat('dataset/S_DL.mat', {"S_DL":S_DL})

    ## ADMM
    mask_2D = mask.reshape(spatial_len,all_bands).T
    nz_idx = np.zeros((173,1))
    nz_idx[0] = 1
    print("start kron...")
    M_idx = np.kron(mask_2D, nz_idx)
    print('end kron...')
    M = M_idx[:172**2, :]
    PtransP = M.reshape(172, 172, -1)
    # matlab(1,2)意思？
    RP_tensor = np.tensordot(E.T, PtransP, axes=1)
    RRtrps_tensor = np.tensordot(E.T, RP_tensor, axes=2)
    savemat('dataset/tensor.mat', {"S_DL":S_DL, "RP_tensor":RP_tensor, 'RRtrps_tensor':RRtrps_tensor})

    X2D_corrupted = X3D_corrupted.reshape(-1, all_bands).T
    # 更新S時後面項帶入Ｒ做化簡後的結果
    RPY = np.zeros((10, 1, 65536))
    for i in range(spatial_len):
        RPY[:,:,i] = np.dot(RP_tensor[:, :, i], X2D_corrupted[:,i].reshape(-1,1))
    RPy = RPY.reshape(-1,1)
    # 更新S時，前面RR'
    RRtrps_per = np.transpose(RRtrps_tensor, (2,0,1))
    #savemat('dataset/tensor.mat', {"RPy":RPy, "RRtrps_per":RRtrps_per})
    # 更新S時，前項I計算
    I = mu/2 * np.eye(N)
    block = np.zeros(RRtrps_per.shape)
    # 更新S時，前項之計算過程
    for i in range(RRtrps_per.shape[0]):
        block[i,:,:] = LA.inv(RRtrps_per[i,:,:].reshape(10, -1) + I)
    block_3D = np.transpose(block, (1,2,0))

    # 儲存暫時結果，給matlab進行視覺化呈現
    savemat('dataset/block_3D.mat', {'block_3D':block_3D})
    # 讀進所產生之S_left.mat
    S_left = loadmat('dataset/S_left.mat')
    S_left = S_left['S_left']
    for i in tqdm(range(50), desc="update s"):
        if i ==0:
            # 初始化S2D及D
            S2D = np.zeros((N,spatial_len))
            D=np.zeros((N,spatial_len))
        # 更新Z
        Z = (1/(mu+lam))*(lam*S_DL+mu*(S2D-D))
        # 更新S時，後面的delta計算
        DELTA = (Z+D)
        delta = DELTA.reshape(-1,1)
        # 更新S時，後項之計算過程
        s_right = RPy + (mu/2)*delta
        # 更新S
        s = S_left@s_right
        S2D = s.reshape((N,65536))   
        # 更新D
        D = D - S2D + Z 
    # 還原影像 ＝ E * S2D 
    X2D_rec=np.dot(E, S2D)
    X3D_rec = X2D_rec.T.reshape(256,256,172)

    print(f"cost time: {datetime.datetime.now()-now}")

    return X3D_rec

def compute_basis(x3dl, N):
    X = x3dl.reshape(-1, x3dl.shape[2])
    M = X.shape[1]
    _, eV = LA.eigh(np.dot(X.T, X))
    E = eV[:,M-N:]

    return E


if __name__=="__main__":
    main()
