import numpy as np
from solver import mesh, solve
from control import solve_control
from cutoff import cutoff

def mesh2(x, t):
    """ Restrict the spatial interval to [1,2] and 
        double the time interval to [0, 2T].
        Return also the index of t=T in the new mesh.
    """
    dt = t[1] - t[0]

    mask = (x >= 1) * (x <= 2)
    x2 = x[mask]

    indices = np.nonzero(mask)[0]
    assert abs(x[indices[0]] - 1) < dt
    assert abs(x[indices[-1]] - 2) < dt

    _, t2 = mesh(L, dt, 2*T)

    indT = np.nonzero(t2 >= T)[0][0]
    assert abs(t2[indT] - T) < dt

    return x2, t2, indT, mask

def d2(v0, v1, v2, dt):
    return (v2 - 2 * v1 + v0) / (dt**2)    

def W(u, f, dx, dt, T):
    res = np.zeros(3)
    for t in range(T-3, T):
        for r in range(T):
            # integral in r is computed by summing manually
            # integrals in s and x are computed by np.sum    
            ind = t - T + 3  
            res[ind] = res[ind] + dt * 0.5 * dt * dx * (
                np.sum(f[t-r]*u[T-r:T+r]-u[t-r]*f[T-r:T+r]))
    return res[0], res[1], res[2]

def Q(u, u0, f, dx, dt, indT):
    wTm2, wTm1, wT = W(u, f, dx, dt, indT)
    w0Tm2, w0Tm1, w0T = W(u0, f, dx, dt, indT)

    uT = u[indT,:]
    u0T = u0[indT,:]
    d2u0T = d2(u0T[:-2], u0T[1:-1], u0T[2:], dx)

    return d2(wTm2, wTm1, wT, dt) - d2(w0Tm2, w0Tm1, w0T, dt) + (
            2 * dx * np.sum(d2u0T*(u0T[1:-1] - uT[1:-1])))
    

if __name__ == '__main__':  
    def q(x):
        return cutoff(x, a=1.4, b=1.6, radius=0.1)
    
    L=3.0
    T=1.5
    dt=0.005
    def I(x):
        return cutoff(x, a=1.2, b=1.8)

    x, t = mesh(L, dt, T)
    f = solve_control(x, t, I)

    # Restrict f in x and extend by zero in t
    x2, t2, indT, mask = mesh2(x, t)
    f2 = np.zeros((len(t2), len(x2)))
    f2[0:len(t),:] = f[:,mask]  

    # Compute u depending on q, and u0 independent from it
    u = solve(x2, t2, f=f2, q=q)
    u0 = solve(x2, t2, f=f2)

    dx = x[1] - x[0]
    Q_comp = Q(u, u0, f2, dx, dt, indT)

    Q_true = dx * np.sum(q(x2))

    print("Error with dt =", dt, "is", abs(Q_comp - Q_true))

