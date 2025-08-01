import numpy as np

def summarize(array: np.ndarray) -> str:
    ''' Return a one line string showing compact data info about the array including:
    - shape
    - dtype
    - min, max, mean, std
    - number of nans
    - number of infs
    '''
    return f'{array.shape} {array.dtype} min={array.min():.4f} max={array.max():.4f} mean={array.mean():.4f} std={array.std():.4f} nans={np.isnan(array).sum()} infs={np.isinf(array).sum()}'
