import numpy as np
import matplotlib.pyplot as plt
from glob import glob

fortran_files = glob('./fortran/output/psi*')
indices = [int(f.split('-')[-1].split('.')[0]) for f in fortran_files]
fortran_files = [fortran_files[i] for i in np.argsort(np.array(indices))]
indices = sorted(indices)

for fort_file,idx in zip(fortran_files,indices):
    fort_data = np.loadtxt(fort_file)

    pyth_data = np.loadtxt(f'./JAX/output/psi-{str(idx).zfill(4)}.txt')
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(fort_data)
    ax1.set_title("fortran")

    ax2.imshow(pyth_data.T)
    ax2.set_title("JAX")

    fig.tight_layout()
    fig.savefig(f'./plots/compareplot-{str(idx).zfill(4)}.png')
    plt.close(fig)
