import numpy as np
from scipy.special import gamma
from scipy.signal import windows
from numba import njit

def _taylor_taper_coeff(SLL: float, NBAR: int, N: int, multiplier = 1):
    t = 10**(abs(SLL/20))
    A = 1/np.pi * np.arccosh(t)
    sig = NBAR/np.sqrt(A**2+(NBAR-0.5)**2)
    n = np.arange(1, NBAR)
    zn = sig*np.sqrt(A**2+(n-0.5)**2)
    z = np.arange(1,NBAR)
    F = np.ones((len(z),))
    for i in range(0,NBAR-1):
        F = F*(1-z**2/zn[i]**2)
    F = F*np.prod(np.arange(1,NBAR))**2 /(gamma(NBAR-z)*gamma(NBAR+z))
    dx = 2/N
    x = np.linspace(-1+dx/2,1-dx/2,N)
    m = np.arange(1, NBAR)
    w = 1 + 2*F @ np.cos(np.pi*(np.outer(m,x)))
    
    array = w / np.max(w)
    array = array * multiplier
    return array


def taylor_taper(SLL: float, NBAR: int, N: int, multiplier: float = 1):
    values = windows.taylor(N, NBAR, SLL)
    return values*multiplier

def chebychev_taper(N: int, attenuation: float):
    return windows.chebwin(N, attenuation)

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def cexp(x,y,z,kx,ky,kz):
    return np.exp(-1j*(x*kx+y*ky+z*kz))


class QuickTaper:

    def __init__(self, function: callable):
        self.function = function

    def __mul__(self, other):
        return self.function(other)

    def __rmul__(self, other):
        return self.function(other)
    
ONES = QuickTaper(lambda x: np.ones((x,)))
TRIANGLE = QuickTaper(lambda x: 1-np.abs(np.linspace(-1,1,x)))
