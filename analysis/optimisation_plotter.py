import numpy as np
import matplotlib.pyplot as plt
from glob import glob

optim_files = glob('./JAX/output/ADpsi*')
indices = [int(f.split('-')[-1].split('.')[0]) for f in optim_files]
optim_files = [optim_files[i] for i in np.argsort(np.array(indices))]
indices = sorted(indices)

optim_data_0 = np.loadtxt('./JAX/output/ADphi.txt')
c_data = np.loadtxt('./JAX/output/ADc.txt')

for optim_file,idx in zip(optim_files,indices):
    optim_data = np.loadtxt(optim_file)

    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(optim_data.T)
    ax2.imshow((optim_data-optim_data_0).T,cmap='coolwarm')
    im3 = ax3.imshow(c_data.T,cmap='jet')
    fig.colorbar(im3,ax=ax3)

    fig.tight_layout()
    fig.savefig(f'./plots/optimplot-{str(idx).zfill(4)}.png')
    plt.close(fig)
