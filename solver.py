import numpy as np
import collections

def solver(I, V, q, f, c, L, dt, C, T):
    x, t = mesh(L, dt, C, T, c)
    return solve(x, t, I, V, q, f, c)

def mesh(L, dt, T, C=1, c=1):
    Nt = int(round(T / dt))             # Number of time steps
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = dt * c / C                     # Step size in space
    Nx = int(round(L / dx))             # Number of steps in space
    x = np.linspace(0, L, Nx + 1)
    return x, t

def solve(x, t, I=None, V=None, q=None, f=None, c=1, callback=None):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    C2 = (c * dt / dx)**2
    Nx = len(x) - 1

    def init_param(p):
        if p is None:
            return np.zeros_like(x)
        elif isinstance(p, collections.abc.Callable):
            return p(x)
        else:
            return p
    I = init_param(I)
    V = init_param(V)
    q = init_param(q)

    # We could also init as f as a spacetime array, but lets save memory
    if f is None:
        f = lambda x, t: 0

    u = np.zeros(Nx + 1)       # solution at new time level
    u_n = np.zeros(Nx + 1)     # solution at 1 time level back
    u_nm1 = np.zeros(Nx + 1)   # solution at 2 time level back

    u_array = []

    # zeroth time step
    u_n[...] = I
    u_array.append(u_n.copy())
    if callback is not None:
        callback(u_n, x, t, 0)

    # special case for first time step
    if isinstance(f, collections.abc.Callable): # f is a function
        ff = f(t[0], x[1:-1])
    else: # f is a numpy array
        ff = f[0, 1:-1]
    u[1:-1] = u_n[1:-1] + dt * V[1:-1] + 0.5 * C2 * (
            u_n[:-2] - 2 * u_n[1:-1] + u_n[2:]) + 0.5 * dt**2 * (
                    q[1:-1] * u_n[1:-1]) + 0.5 * dt**2 * ff
    u[0] = 0
    u[Nx] = 0

    u_array.append(u.copy())
    if callback is not None:
        callback(u, x, t, 1)

    u_nm1[:] = u_n
    u_n[:] = u

    # rest of the time steps
    for n in range(1, len(t) - 1):
        if isinstance(f, collections.abc.Callable): # f is a function
            ff = f(t[n], x[1:-1])
        else: # f is a numpy array
            ff = f[n, 1:-1]

        u[1:-1] = -u_nm1[1:-1] + 2 * u_n[1:-1] + C2 * (
                u_n[:-2] - 2 * u_n[1:-1] + u_n[2:]) + dt**2 * (
                    q[1:-1] * u_n[1:-1]) + dt**2 * ff
        # set boundary to zero
        u[0] = 0
        u[Nx] = 0

        u_array.append(u.copy())
        if callback is not None:
            if callback(u, x, t, n + 1):
                break

        u_nm1[:] = u_n
        u_n[:] = u

    return np.array(u_array)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from numpy import sin, cos, power, pi, e

    u = lambda t, x: power(e,t)*sin(2*pi*x)
    f = lambda t, x: power(e,t)*(4*power(pi,2) - x)*sin(2*pi*x)
    ut = lambda t, x: power(e,t)*sin(2*pi*x)
    ux = lambda t, x: 2*power(e,t)*pi*cos(2*pi*x)
    q = lambda x: 1 + x
    
    L=3.0
    T=1.5
    dt=0.01
    
    x, t = mesh(L, dt, T)
    u_comp = solve(x, t, I=u(0, x), V=ut(0, x), q=q, f=f)

    u_compT = u_comp[-1,:]

    plt.plot(x, u_compT, label='computed u')
    plt.plot(x, u(T, x), label='exact u')
    plt.legend()
    plt.show()

    print(np.max(np.abs(u(T, x) - u_compT)))

