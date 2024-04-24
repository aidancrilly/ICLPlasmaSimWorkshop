from lax_solver import *
import numpy as np
import time

def c_uniform(xx,yy):
    return jnp.ones_like(xx)

def init_gaussian(x,y,sig,cfunc):
    xx,yy = jnp.meshgrid(x,y,indexing='ij')
    dist = jnp.sqrt(xx**2+yy**2)
    c = cfunc(xx,yy)
    r = -c*xx*jnp.exp(-0.5*dist**2/sig**2)/sig**2
    l = -c*yy*jnp.exp(-0.5*dist**2/sig**2)/sig**2
    s = jnp.zeros_like(xx)
    psi = jnp.exp(-0.5*dist**2/sig**2)
    return (r,l,s,psi),c

def for_loop_runner(u_0,args,IO=False):
    jit_Lax = generate_Lax_Step(args)

    u = jit_Lax(u_0)
    u = u_0
    start = time.time()
    for i in range(1000):
        u = jit_Lax(u)
        if(i%10 == 0 and IO):
            np.savetxt(f'./JAX/output/psi-{str(i).zfill(4)}.txt',u[-1])
    finish = time.time()
    print(f'time elapsed: {finish-start}')

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

for_loop_runner(u_0,args)