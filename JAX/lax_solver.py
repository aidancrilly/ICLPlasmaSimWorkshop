import jax
import jax.numpy as jnp

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

def get_ghost(f):
    f_x     = jnp.concatenate((f[0:1,:],f,f[-1:,:]),axis=0)
    f_ghost = jnp.concatenate((f_x[:,0:1],f_x,f_x[:,-1:]),axis=1)
    return f_ghost

def Lax_Update(f,jx,jy):
    f_1 = 0.25 * (f[:-2,1:-1] + f[2:,1:-1] + f[1:-1,:-2] + f[1:-1,2:])
    f_1 -= 0.5 * (jx[2:,:] - jx[:-2,:] + jy[:,2:] - jy[:,:-2])
    return f_1

def generate_Lax_Step(args):
    dt = args['dt']
    delta = args['delta']
    c = args['c']
    Nx,Ny = args['Nx'],args['Ny']
    def Lax_Step(u):
        r_0,l_0,s_0,psi_0 = u

        # Reflective boundaries
        r_0_ghost = get_ghost(r_0)
        l_0_ghost = get_ghost(l_0)
        s_0_ghost = get_ghost(s_0)
        c_ghost   = get_ghost(c)

        Fr_x = -dt/delta*c_ghost[:,1:-1]*s_0_ghost[:,1:-1]
        Fr_y = jnp.zeros((Nx,Ny+2))
        r_1 = Lax_Update(r_0_ghost,Fr_x,Fr_y)
        # Reflective boundaries
        r_1 = r_1.at[0,:].set(0.0)
        r_1 = r_1.at[-1,:].set(0.0)

        Fl_x = jnp.zeros((Nx+2,Ny))
        Fl_y = -dt/delta*c_ghost[1:-1,:]*s_0_ghost[1:-1,:]
        l_1 = Lax_Update(l_0_ghost,Fl_x,Fl_y)
        # Reflective boundaries
        l_1 = l_1.at[:,0].set(0.0)
        l_1 = l_1.at[:,-1].set(0.0)

        Fs_x = -dt/delta*c_ghost[:,1:-1]*r_0_ghost[:,1:-1]
        Fs_y = -dt/delta*c_ghost[1:-1,:]*l_0_ghost[1:-1,:]
        s_1 = Lax_Update(s_0_ghost,Fs_x,Fs_y)

        psi_1 = psi_0 + 0.5 * dt * (s_0 + s_1)

        u = r_1,l_1,s_1,psi_1

        return u
    
    return jax.jit(Lax_Step)