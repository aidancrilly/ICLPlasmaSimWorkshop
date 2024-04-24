from lax_solver import *
import numpy as np
import time

def for_loop_runner(u_0,args,IO=False,Nt=1000,prefix='psi'):
    jit_Lax = generate_Lax_Step(args)

    u = jit_Lax(u_0)
    u = u_0
    start = time.time()
    for i in range(Nt):
        u = jit_Lax(u)
        if(i%10 == 0 and IO):
            np.savetxt(f'./JAX/output/{prefix}-{str(i).zfill(4)}.txt',u[-1])
    finish = time.time()
    print(f'time elapsed: {finish-start}')

if __name__ == "__main__":

    delta = 0.01
    Nx = 200
    Ny = 300
    x = delta*jnp.arange(Nx)
    y = delta*jnp.arange(Ny)
    x -= jnp.mean(x)
    y -= jnp.mean(y)
    sig = 10*delta

    u_0,c = init_gaussian(x,y,sig,c_uniform)
    dtmult = 0.5

    args = {'delta' : delta, 'dt' : dtmult*delta/c.max(), 'c' : c, 'Nx' : Nx, 'Ny' : Ny}

    for_loop_runner(u_0,c)