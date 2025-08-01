import matplotlib.pyplot as plt
import numpy as np
import time

from src.optycal.antennas.interpolator import create_interpolator, intmode

def plot_interpolation_results(xs, ys, Ftrue, Finterp):
    """
    Creates a 3-panel plot:
    - Left: Ftrue
    - Middle: Finterp
    - Right: abs(Ftrue - Finterp)
    
    Parameters
    ----------
    xs : 1D array of shape (Nx,)
        x-axis coordinates
    ys : 1D array of shape (Ny,)
        y-axis coordinates
    Ftrue : 2D array of shape (Nx, Ny)
        Ground truth function values
    Finterp : 2D array of shape (Nx, Ny)
        Interpolated values at the same grid
    """
    error = np.abs(Ftrue - Finterp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    plots = [
        (Ftrue, "True Function", axes[0]),
        (Finterp, "Interpolated", axes[1]),
        (error, "Absolute Error", axes[2]),
    ]

    for data, title, ax in plots:
        mesh = ax.pcolormesh(xs, ys, data)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(mesh, ax=ax)

    plt.show()



# --- Your inputs ---
xin = np.linspace(0, 10, 51)
yin = np.linspace(0, 10, 51)
X, Y = np.meshgrid(xin, yin, indexing='ij')


xsample = np.linspace(0, 10, 2001)
ysample = np.linspace(0, 10, 2001)
XS, YS = np.meshgrid(xsample, ysample, indexing='ij')

print(X[:,0])
print(X[0,:])
Z = np.cos(3 * X) * np.sin(2 * Y)
# --- Time true function evaluation ---
start_true = time.perf_counter()
Ztrue = np.cos(3 * XS) * np.sin(2 * YS)
end_true = time.perf_counter()
print(f"Time to compute Z (true values): {end_true - start_true:.6f} seconds")

# --- Compute interpolator matrix (assumed fast / precomputation) ---
interp = create_interpolator(xin, yin, Z, intmode.NaturalSpline)

# --- Time interpolation ---
start_interp = time.perf_counter()
Zinterp = interp(XS, YS)
end_interp = time.perf_counter()
print(f"Time to compute Zinterp (interpolated): {end_interp - start_interp:.6f} seconds")

# --- Plot results ---

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(xsample, Zinterp[1000, :], label='Interpolated')
ax.plot(xin, Z[25,:], label='True')
plt.show()
plot_interpolation_results(xsample, ysample, Ztrue, Zinterp)