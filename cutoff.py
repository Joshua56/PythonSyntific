import numpy as np
from scipy.interpolate import KroghInterpolator

def cutoff(x, a=1, b=2, radius=0.2, L=3, deg=3):
    a0 = a
    b0 = b
    a1 = a0 + radius
    b1 = b0 - radius

    assert a1 <= b1, 'a + radius must be less than b - radius'

    pts1 = np.concatenate((a0*np.ones(deg), a1*np.ones(deg)))
    vals1 = np.zeros(2*deg)
    vals1[deg] = 1
    krogh1 = KroghInterpolator(pts1, vals1)

    pts2 = np.concatenate((b1*np.ones(deg), b0*np.ones(deg)))
    vals2 = np.zeros(2*deg)
    vals2[0] = 1
    krogh2 = KroghInterpolator(pts2, vals2)

    return np.piecewise(x, [
        np.logical_and(x > a0, x < a1),
        np.logical_and(x >= a1, x <= b1),
        np.logical_and(x > b1, x < b0)
        ], [krogh1, 1, krogh2, 0])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 3, 200)
    y = cutoff(x)
    plt.plot(x, y)
    plt.show()
