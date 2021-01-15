from solver import solver, solve_f, solve_Q
from cutoff import cutoff
import sys
import numpy as np



def W(u, f, dx, dt):
    Nt = u.shape[0]
    T= Nt//2
    res = np.zeros(T)
    for t in range(T):
        for r in range(T):
            res[t] = 0.5 * dt * dx * np.sum(f[t-r]*u[T-r:T+r]-u[t-r]*f[T-r:T+r])
    return np.cumsum(res*dt)


def int_w(u: np.ndarray, f: np.ndarray) -> np.ndarray:
    r"""Calculates W(s, t)
        W(s, t) = \int_0^1 u(x, t) * f(x, s) dx
    """
    assert u.shape == f.shape
    return np.dot(u, f.T).T


class Task6:
    def __init__(self, m=2, L=3.0, T=2, dt=0.01, xi=.15):
        self.m = m
        self.L = L
        self.T = T
        self.dt = dt
        self.xi = xi
        self.C = 0.9
        self.x, self.t = self.get_x_t()

    def u_exact(self, x, t):
        return np.cos(self.m * np.pi / self.L * t) * np.sin(self.m * np.pi / self.L * x)

    def get_x_t(self):
        """Dummy solver call just to get the x & t"""
        _, _, x, t, _ = solver(I=0, V=0, f=0, q=0, c=1, L=self.L, dt=self.dt, C=self.C, T=self.T)
        return x, t

    def get_cuttoff_func(self, a, b):
        """In interval (a, b) value is 1 elsewhere 0"""
        _, y = cutoff(self.x, a=a, b=b, L=self.L)
        return y

    def get_I(self, eta):
        return lambda x: (eta * np.exp(1j * self.xi * self.x)).real  # only getting the real part

    def I(self, x):
        return self.get_cuttoff_func(a=1.2, b=1.8)


    def compute_f(self):
        X = self.get_cuttoff_func(a=1, b=2)
        q = self.get_cuttoff_func(a=1.4, b=1.6)

        u_array, _, _, _, _ = solver(
            I=self.I,
            V=0,
            f=0,
            q=q,
            c=1,
            L=self.L,
            dt=self.dt,
            C=self.C,
            T=self.T
        )

        #  import matplotlib.pyplot as plt
        #  plt.plot(self.x, q, label='q')
        #  plt.plot(self.x, self.I(self.x), label='eta')
        #  plt.plot(self.x, self.get_I(self.I(self.x))(self.x), label='exp')
        #  plt.legend()
        #  plt.show()

        u = np.array(u_array)
        v = (X * u)[::-1]  # reverse the array
        f = solve_f(v=v, dx=self.x[1] - self.x[0], dt=self.dt)
        return f

    def compute_u(self):
        f = self.compute_f()
        u, _, _, _, _ = solver(
            I=0,
            V=0,
            q=self.get_cuttoff_func(a=1.4, b=1.6),  # using non zero q
            f=f,
            c=1,
            L=self.L,
            dt=self.dt,
            C=self.C,
            
            ######################################
            # T=2*self.T
                   
            T=(self.T)/2
            ######################################
        )
        u = np.array(u)
        return u

    def compute_u0(self):
        f = self.compute_f()
        u0, _, _, _, _ = solver(
            I=0,
            V=0,
            q=0,
            f=f,
            c=1,
            L=self.L,
            dt=self.dt,
            C=self.C,
            T=(self.T)/2
        )
        u0 = np.array(u0)
        return u0

    def compute_W(self):
        u = self.compute_u()
        f = self.compute_f()
        
      
        ######################################
        # w = W(u, f, u.shape[0] // 2, self.x[1] - self.x[0], self.dt)
        w = W(u, f, self.x[1] - self.x[0], self.dt)
        ######################################        
        return w

    def compute_W0(self):
        u0 = self.compute_u0()
        f = self.compute_f()

     
        ######################################
        # w = W(u0, f, u0.shape[0] // 2, self.x[1] - self.x[0], self.dt)
        w = W(u0, f, self.x[1] - self.x[0], self.dt)                
        ######################################
        return w

    def compute_Q(self):
        w = self.compute_W()
        w0 = self.compute_W0()
        Q = solve_Q(w - w0, dt=self.dt) + self.xi ** 2
        u = self.compute_u()
        u0 = self.compute_u0()
        Q_2nd_part = solve_Q(u0, dt=self.x[1] - self.x[0])
        Q_2nd_part = Q_2nd_part[-2] + self.xi ** 2 * u0[-1]
        Q_2nd_part *= (u0[-1] - u[-1])/2

        Q =  np.sum(2 * Q_2nd_part * (self.x[1] - self.x[0]))
        return Q

    def compute_fourier_transform(self):
        return np.sum(self.I(self.x))  * (self.x[1] - self.x[0])

    def verify(self):
        Q = self.compute_Q()
        fourier = self.compute_fourier_transform()
        print('Difference:', np.abs(Q - fourier))


if __name__ == '__main__':
    task6 = Task6(dt=0.01)
    task6.verify()
