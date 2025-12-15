import numpy as np
from scipy.signal import windows as win

def _to_complex(w):
    """Convert to complex128 array."""
    return np.asarray(w, dtype=np.complex128)

def ones(n):
    return np.ones(n)

def taylor(n, nbar=4, sll=30.0, sym=True):
    """Taylor window.

    Args:
        n (int): Number of points.
        nbar (int, optional): Number of nearly constant-level sidelobes. Defaults to 4.
        sll (float, optional): Desired sidelobe level in dB. Defaults to 30.0.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Taylor taper.
    """
    return _to_complex(win.taylor(n, nbar=nbar, sll=sll, norm=False, sym=sym))

def chebwin(n, at=100.0, sym=True):
    """Dolph–Chebyshev window.

    Args:
        n (int): Number of points.
        at (float, optional): Attenuation in dB. Defaults to 100.0.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Chebyshev taper.
    """
    return _to_complex(win.chebwin(n, at=at, sym=sym))

def kaiser(n, beta=14.0, sym=True):
    """Kaiser window.

    Args:
        n (int): Number of points.
        beta (float, optional): Shape parameter. Defaults to 14.0.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Kaiser taper.
    """
    return _to_complex(win.kaiser(n, beta=beta, sym=sym))

def hann(n, sym=True):
    """Hann window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Hann taper.
    """
    return _to_complex(win.hann(n, sym=sym))

def hamming(n, sym=True):
    """Hamming window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Hamming taper.
    """
    return _to_complex(win.hamming(n, sym=sym))

def blackman(n, sym=True):
    """Blackman window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Blackman taper.
    """
    return _to_complex(win.blackman(n, sym=sym))

def blackmanharris(n, sym=True):
    """Blackman–Harris window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Blackman–Harris taper.
    """
    return _to_complex(win.blackmanharris(n, sym=sym))

def nuttall(n, sym=True):
    """Nuttall window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Nuttall taper.
    """
    return _to_complex(win.nuttall(n, sym=sym))

def flattop(n, sym=True):
    """Flat top window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued flat top taper.
    """
    return _to_complex(win.flattop(n, sym=sym))

def bartlett(n, sym=True):
    """Bartlett (triangular) window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Bartlett taper.
    """
    return _to_complex(win.bartlett(n, sym=sym))

def triang(n, sym=True):
    """Triangular window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued triangular taper.
    """
    return _to_complex(win.triang(n, sym=sym))

def boxcar(n):
    """Rectangular (boxcar) window.

    Args:
        n (int): Number of points.

    Returns:
        np.ndarray: Complex-valued rectangular taper.
    """
    return _to_complex(win.boxcar(n))

def gaussian(n, std=None, sym=True):
    """Gaussian window.

    Args:
        n (int): Number of points.
        std (float, optional): Standard deviation. Defaults to 0.4*(n-1)/2.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Gaussian taper.
    """
    if std is None:
        std = 0.4 * (n - 1) / 2.0
    return _to_complex(win.gaussian(n, std=std, sym=sym))

def general_gaussian(n, p=2.0, sig=None, sym=True):
    """Generalized Gaussian window.

    Args:
        n (int): Number of points.
        p (float, optional): Power. Defaults to 2.0.
        sig (float, optional): Standard deviation. Defaults to 0.4*(n-1)/2.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued generalized Gaussian taper.
    """
    if sig is None:
        sig = 0.4 * (n - 1) / 2.0
    return _to_complex(win.general_gaussian(n, p=p, sig=sig, sym=sym))

def cosine(n, sym=True):
    """Cosine window.

    Args:
        n (int): Number of points.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued cosine taper.
    """
    return _to_complex(win.cosine(n, sym=sym))

def tukey(n, alpha=0.5, sym=True):
    """Tukey (tapered cosine) window.

    Args:
        n (int): Number of points.
        alpha (float, optional): Shape parameter. Defaults to 0.5.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued Tukey taper.
    """
    return _to_complex(win.tukey(n, alpha=alpha, sym=sym))

def dpss(n, NW=2.5, Kmax=1, sym=True):
    """Discrete prolate spheroidal sequence (Slepian) window.

    Args:
        n (int): Number of points.
        NW (float, optional): Time-halfbandwidth product. Defaults to 2.5.
        Kmax (int, optional): Number of tapers to return. Defaults to 1.
        sym (bool, optional): Symmetric or periodic. Defaults to True.

    Returns:
        np.ndarray: Complex-valued DPSS taper (first if Kmax > 1).
    """
    w = win.dpss(n, NW=NW, Kmax=Kmax, sym=sym)
    if w.ndim == 2:
        w = w[0]
    return _to_complex(w)

def uniform(n: int):
    """Simple uniform taper

    Args:
        n (_type_): _description_
    """
    return np.ones((n,))