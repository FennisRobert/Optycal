import numpy as np
from scipy.special import gamma
from scipy.signal import windows
from numba import njit

def _taylor_taper_coeff(SLL: float, NBAR: int, N: int, multiplier = 1):
    """Calculates the Taylor taper coefficients.

    Args:
        SLL (float): The sidelobe level.
        NBAR (int): The number of non-zero elements in the taper.
        N (int): The total number of elements in the taper.
        multiplier (int, optional): A multiplier for the taper coefficients. Defaults to 1.

    Returns:
        np.ndarray: The Taylor taper coefficients.
    """
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
    """
    Creates a Taylor taper window.

    Args:
        SLL (float): The sidelobe level.
        NBAR (int): The number of non-zero elements in the taper.
        N (int): The total number of elements in the taper.
        multiplier (float, optional): A multiplier for the taper coefficients. Defaults to 1.

    Returns:
        np.ndarray: The Taylor taper coefficients.
    """
    values = windows.taylor(N, NBAR, SLL)
    return values * multiplier

def chebychev_taper(N: int, attenuation: float):
    """Creates a Chebyshev taper window.

    Args:
        N (int): The total number of elements in the taper.
        attenuation (float): The desired attenuation level in dB.

    Returns:
        np.ndarray: The Chebyshev taper coefficients.
    """
    return windows.chebwin(N, attenuation)

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def cexp(x, y, z, kx, ky, kz):
    return np.exp(-1j * (x * kx + y * ky + z * kz))


class QuickTaper:
    def __init__(self, function: callable):
        self.function = function

    def __mul__(self, other):
        return self.function(other)

    def __rmul__(self, other):
        return self.function(other)
    
ONES = QuickTaper(lambda x: np.ones((x,)))
TRIANGLE = QuickTaper(lambda x: 1-np.abs(np.linspace(-1,1,x)))
HANN = QuickTaper(lambda x: windows.hann(x))
HAMMING = QuickTaper(lambda x: windows.hamming(x))
BLACKMAN = QuickTaper(lambda x: windows.blackman(x))
FLATTOP = QuickTaper(lambda x: windows.flattop(x))
BARTLETT = QuickTaper(lambda x: windows.bartlett(x))
