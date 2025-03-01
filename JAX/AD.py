import diffrax
from lax_solver import *
from runner import *
from utils import *

class Stepper(diffrax.Euler):
    """

    :param cfg:
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        args['dt'] = t1 - t0
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful

def solve_over_T_with_c(c,u_0,T,args):
    args = {'delta' : args['delta'], 'dt' : args['dtmult']*args['delta']/c.max(), 'c' : c, 'Nx' : args['Nx'], 'Ny' : args['Ny']}
    jit_Lax = generate_Lax_Step(args)
    def diffrax_Lax_interface(t,y,args):
        y_0 = y.reshape(4,args['Nx'],args['Ny'])
        y_0 = (y_0[0,:,:],y_0[1,:,:],y_0[2,:,:],y_0[3,:,:])

        y_1 = jit_Lax(y_0)

        y_1 = jnp.vstack(y_1)
        y_1 = y_1.flatten()

        return y_1

    term = diffrax.ODETerm(diffrax_Lax_interface)
    y0 = jnp.vstack(u_0).flatten()

    saveat = diffrax.SaveAt(ts=jnp.array([0.0,T]))
    solution = diffrax.diffeqsolve(term, Stepper(), t0=0, t1=T, dt0=args['dt'], y0=y0, saveat = saveat, args = args, adjoint=diffrax.RecursiveCheckpointAdjoint())

    ys = solution.ys.reshape(2,4,args['Nx'],args['Ny'])

    psi = ys[-1,-1,:,:]
    return psi

def all_dpsiTdc(c,u_0,T,args):
    dpsidTdc = jax.jacrev(solve_over_T_with_c,argnums=0)(c,u_0,T,args)
    return dpsidTdc

def MSE(c,phi,u_0,T,args):
    psi = solve_over_T_with_c(c,u_0,T,args)
    return jnp.sum((psi-phi)**2)
    
def value_and_grad_MSE(c,phi,u_0,T,args):
    dMSEdc = jax.value_and_grad(MSE,argnums=0)
    return dMSEdc(c,phi,u_0,T,args)

def speed_func(coeffs,args):
    Nx,Ny = args['Nx'],args['Ny']
    c_ones = jnp.ones((Nx,Ny))
    xx,yy = args['xx'],args['yy']
    x_norm = xx/jnp.amax(xx)
    y_norm = yy/jnp.amax(yy)
    x_l = legendre_recurrence(x_norm,coeffs.shape[0])[1:,:,:]
    y_l = legendre_recurrence(y_norm,coeffs.shape[1])[1:,:,:]
    delta_c = jnp.einsum('ij,ikl,jkl->kl',coeffs,x_l,y_l)
    return c_ones+delta_c

def MSE_w_regulariser(c_coeffs,phi,u_0,T,args):
    c_smoothed = speed_func(c_coeffs,args)
    psi = solve_over_T_with_c(c_smoothed,u_0,T,args)
    return jnp.sum((psi-phi)**2)
    
def value_and_grad_MSE_w_regulariser(c_coeffs,phi,u_0,T,args):
    dMSEdc = jax.value_and_grad(MSE_w_regulariser,argnums=0)
    return dMSEdc(c_coeffs,phi,u_0,T,args)

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

    args = {'delta' : delta, 'dtmult' : dtmult, 'c' : c, 'Nx' : Nx, 'Ny' : Ny}

    T = 2.0
    xx,yy = jnp.meshgrid(x,y,indexing='ij')
    sig_phi = 20*delta
    phi = 0.25*(jnp.exp(-0.5*(xx**2+(yy-1.0)**2)/sig_phi**2)+jnp.exp(-0.5*(xx**2+(yy+1.0)**2)/sig_phi**2))

    mag = 0.01
    coeffs = mag*(np.random.rand(4,6)-0.5)
    args['xx'] = xx
    args['yy'] = yy

    np.savetxt('./JAX/output/ADphi.txt',phi)

    # Simple gradient descent algorithm
    n_iter = 500
    learning_rate = 1e-5
    for iter in range(n_iter):
        loss,dLdc = value_and_grad_MSE_w_regulariser(coeffs,phi,u_0,T,args)
        print(iter,loss)
        coeffs = coeffs-learning_rate*dLdc

        c_out = speed_func(coeffs,args)
        np.savetxt('./JAX/output/ADc.txt',c_out)

    dt = dtmult*delta/c_out.max()
    Nt = int(np.ceil(T/dt))
    args = {'delta' : delta, 'dt' : dt, 'c' : c_out, 'Nx' : Nx, 'Ny' : Ny}

    for_loop_runner(u_0,args,IO=True,Nt=Nt,prefix='ADpsi')
