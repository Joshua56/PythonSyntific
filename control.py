import numpy as np
from solver import mesh, solve
from cutoff import cutoff

def solve_control(x, t, I, X=None):
    if X is None:
        X = cutoff(x, a=1, b=2)

    u = solve(x, t, I=I)
    v = X * u
    v = v[::-1] # flip because v(x, t) = X(x) * u(x, T-t)

    dx = x[1] - x[0]
    dt = t[1] - t[0]
    return box(v, dx, dt)

def box(v, dx, dt):
    """ Compute the action of the wave operator on v, that is,
        \Box v = d^2v/dt^2 - d^2v/dx^2 
    """
    Nt, Nx = v.shape
    f_array = []
    f_array.append(np.zeros(Nx))
    f = np.zeros(Nx)
    for n in range(1, Nt - 1):
        f[1:-1] = (v[n + 1, 1:-1] - 2 * v[n, 1:-1] + v[n - 1, 1:-1]) / (dt**2) - \
            (v[n, 2:] - 2 * v[n, 1:-1] + v[n, :-2]) / (dx**2)
        f_array.append(f.copy())
    f_array.append(np.zeros(Nx))
    return np.array(f_array)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    L=3.0
    T=1.5
    dt=0.01
    def I(x):
        return cutoff(x, a=1.2, b=1.8)

    x, t = mesh(L, dt, T)
    f = solve_control(x, t, I)

    # Test here also restriction in x and extension in t

    mask = (x >= 1) * (x <= 2)
    indices = np.nonzero(mask)[0]
    assert abs(x[indices[0]] - 1) < dt
    assert abs(x[indices[-1]] - 2) < dt

    # Restrict the spatial interval to interval [1,2]
    x2 = x[mask]
    # Double the time interval to [0, 2T]
    _, t2 = mesh(L, dt, 2*T)

    # Restrict f in x and extend by zero in t
    f2 = np.zeros((len(t2), len(x2)))
    f2[0:len(t),:] = f[:,mask]  

    mask = t2 >= T
    indT = np.nonzero(mask)[0][0]
    assert abs(t2[indT] - T) < dt

    u = solve(x2, t2, f=f2)
    uT = u[indT,:]

    plt.plot(x2, uT, label='u using f')
    plt.plot(x2, I(x2), label='target')
    plt.legend()
    plt.show()

    #plt.imshow(f2)
    #plt.show()

    print(np.max(np.abs(I(x2) - uT)))

