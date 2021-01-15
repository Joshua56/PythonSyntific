from solver import solver, solve_f, solve_Q
from cutoff import cutoff
import sys
import numpy as np
 

class Task5:
    def __init__(self, m=2, L=3.0, T=1.5, dt=0.01):
        self.m = m
        self.L = L
        self.T = T
        self.dt = dt
        self.a = self.L/2 - 0.5
        self.b = self.L/2 + 0.5
        self.C = 0.9
        self.x = self.get_x()

    def get_x(self):
        """Dummy solver call just to get the x"""
        _, _, x, _, _ = solver(I=0, V=0, f=0, q=0, c=1, L=self.L, dt=self.dt, C=self.C, T=self.T)
        return x

    def get_cuttoff_func(self):
        """In interval (a, b) value is 1 elsewhere 0"""
        _, y = cutoff(self.x, a=self.a, b=self.b, L=self.L)
        return y

    def u_exact(self, x, t):
        return np.cos(self.m * np.pi / self.L * t) * np.sin(self.m * np.pi / self.L * x)

    def I(self, x):
        value = self.u_exact(x, 0)
        a, b = self.a, self.b
        if isinstance(value, np.ndarray):
            value[(x < a) | (x > b)] = 0
        else:
            if not (a < x < b):
                value = 0
        return value

    def compute_u(self):
        u_array, _, _, _, _ = solver(
            I=self.I,
            V=0,
            f=0,
            q=0,
            c=1,
            L=self.L,
            dt=self.dt,
            C=0.9,
            T=self.T
        )
        return np.array(u_array)

    def compute_f(self):
        X = self.get_cuttoff_func()
        u = self.compute_u()

        v = (X * u)[::-1]  # reverse the array
        f = solve_f(v=v, dx=self.x[1] - self.x[0], dt=self.dt)
        return f

    def compute_u_using_f(self):
        f = self.compute_f()
        _, u, _, _, _ = solver(
            I=0,
            V=0,
            q=0,
            f=f,
            c=1,
            L=self.L,
            dt=self.dt,
            C=self.C,
            T=self.T
        )
        return u

    def verify(self):
        X = self.get_cuttoff_func()
        u = self.compute_u_using_f()
        EPS = 1 / 10
        print('Solution given by the sover is close to XI at T: ',
            np.alltrue(np.abs((X * self.u_exact(self.x, 0)) - u) < EPS))


if __name__ == '__main__':
    task5 = Task5()
    task5.verify()
