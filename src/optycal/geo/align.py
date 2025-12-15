import numpy as np


def AlignX(x: np.ndarray,y: np.ndarray,z: np.ndarray):
    return np.array([1,0,0])

def AlignY(x: np.ndarray,y: np.ndarray,z: np.ndarray):
    return np.array([0,1,0])

def AlignZ(x: np.ndarray,y: np.ndarray,z: np.ndarray):
    return np.array([0,0,1])

def AlignOrigin(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    return np.ndarray([x,y,z])/(x**2+y**2+z**2)**0.5