import optycal as opt
import numpy as np

"""Example 3: Steered 2D Patch Array in a Rotated Coordinate System

In this example we:
- Define the same 2D patch array as before.
- Mount the array in a rotated coordinate system (tilt and azimuth).
- Apply electronic beam steering using set_scan_direction.
- Query a single element pattern for inspection.
- Evaluate and visualize the steered far-field on cuts and on a 3D sphere.

"""

# Operating frequency [Hz]
f = 3.1e9

# Speed of light in vacuum [m/s]
c0 = 299792458

# Wavelength [m]
wavelength = c0 / f

# Element spacing [m], 0.55λ to limit grating lobes
ds = 0.55 * wavelength

# Define a rotated coordinate system CS2 based on the global coordinate system:
# - copy(): Creates an inherited coordinate system that is equivalent to the original.
# - rotate_basis((0, -1, 0), 15, True): tilt 15 deg around axis (0, -1, 0) in degrees
# - rotate_basis((0, 0, 1), 10): rotate 10 deg around z-axis for azimuthal orientation
# This represents a physically tilted and rotated array mount.
CS2 = opt.GCS.copy().rotate_basis((0, -1, 0), 15, True).rotate_basis((0, 0, 1), 10)

# Create antenna array in the rotated coordinate system CS2.
array = opt.AntennaArray(f, CS2)

# Add a 2D patch array:
# - 20 elements along y with Taylor taper (N=20, nbar=5, SLL=-30 dB)
# - 10 elements along z with uniform taper
# - Spacing ds in y and z
# - Initial phase 0
# - Patch element patterns for near-field and far-field
array.add_2d_array(
    opt.taper.taylor(20, 5, 30),
    opt.taper.uniform(10),
    (0, ds, 0),
    (0, 0, ds),
    0,
    opt.patch_pattern_nf,
    opt.patch_pattern_ff,
)

# Set electronic scan direction:
# Arguments: (theta_scan, phi_scan) in degrees in the array's coordinate system.
# Here (90, 0) steers the main beam relative to the rotated CS2.
# The scan direction is always in global coordinates. theta=0 is the +Z direction so theta=90, phi=0 is the +X direction.
array.set_scan_direction(90, 0)

# Define 1D far-field cuts with 1 degree resolution:
azi, ele = opt.FF1D.aziele(dangle=1)

# Define 2D spherical sampling grid with 1 degree resolution:
sphere = opt.FF2D.sphere(1)

# Compute and attach array far-field on all datasets:
array.expose_ff(azi)
array.expose_ff(ele)
array.expose_ff(sphere)

# Plot normalized far-field cuts in dB:
# Normalized by Eiso for directivity-type visualization.
opt.plot_ff(azi.phi, azi.field.normE / opt.Eiso, dB=True)
opt.plot_ff(ele.theta, ele.field.normE / opt.Eiso, dB=True)

# Set up 3D Optycal display:
display = opt.OptycalDisplay()

# Add 3D pattern surface:
# - 'Etheta' magnitude in dB
# - Floor at -30 dB
# - rmax=0.5 to keep lobes compact and emphasize relative levels.
display.add_surf(*sphere.surfplot('Etheta', 'abs', dB=True, dBfloor=-30, rmax=0.5))

# Add a visual representation of the array geometry (from CS2) to the display.
display.add_array_object(array)

# Render interactive 3D view with array and steered pattern.
display.show()
