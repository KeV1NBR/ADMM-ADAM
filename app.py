from scipy.io import loadmat, savemat
import numpy as np
from PIL import Image

def main():
    # load dataset
    x3dl = loadmat('dataset/X3DL.mat')
    mask = loadmat('dataset/mask.mat')
    ottawa = loadmat('dataset/Ottawa.mat')
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

    # plot raw image
    print(x3dl['X3D_DL'][0])
    plot(x3dl['X3D_DL'], "DL")

def plot(data, file_name, RGB_channel=[18, 8, 2]):
    img = Image.fromarray(np.uint8(data[:, :, RGB_channel] * 255))
    img.save(f"reports/{file_name}.jpg")

if __name__=="__main__":
    main()
