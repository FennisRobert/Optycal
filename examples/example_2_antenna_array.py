import optycal as opt

"""Example 2: 2D Patch Array with Tapering

In this example we simulate a 2D patch antenna array using Optycal.
We:
- Define the operating frequency and element spacing.
- Build a rectangular 20 x 10 array with specified tapers.
- Compute the far-field in 1D cuts and on a full sphere.
- Visualize both the cuts and the 3D radiation pattern.

Please notice that because we don't use embedded element patters, that the resulting 
far-field may not conserve power in quite the way that we expact. In order to do that we 
would need the exact embedded element patterns of our antenna array.

To compensate, Optycal performs a power conservation integral to limit the total radiated power to 1W.
"""

# Operating frequency [Hz]
f = 3.1e9

# Speed of light in vacuum [m/s]
c0 = 299792458

# Wavelength corresponding to the operating frequency
wavelength = c0 / f

# Element spacing [m], here 0.55λ to control grating lobes
ds = 0.55 * wavelength

# Create an empty antenna array at the given frequency in global coordinate system.
# Individual elements and layout are added via helper methods.
array = opt.AntennaArray(f, opt.GCS)

# Define a 2D array:
# - In the "y" direction: 20 elements with Taylor taper (N=20, nbar=5, SLL=-30 dB)
# - In the "z" direction: 10 elements with uniform taper
# - Element spacing: (0, ds, 0) between elements along y, (0, 0, ds) along z
# - Row offset: 0 (0.5 would be hexagonal array)
# - Element patterns: patch_pattern_nf / patch_pattern_ff used for NF/FF of each element
array.add_2d_array(
    opt.taper.taylor(20, 5, 30),
    opt.taper.uniform(10),
    (0, ds, 0),
    (0, 0, ds),
    0,
    opt.patch_pattern_nf,
    opt.patch_pattern_ff,
)

# Create 1D far-field datasets:
# - azi: azimuth cut (phi-scan) at fixed elevation
# - ele: elevation cut (theta-scan) at fixed azimuth
# dangle=1 gives 1 degree sampling.
azi, ele = opt.FF1D.aziele(dangle=1)

# Create a full 2D spherical far-field sampling grid with 1 degree resolution.
sphere = opt.FF2D.sphere(1)

# Expose the far-field of the array to each dataset.
# This computes and stores the field values directly in azi.field, ele.field, sphere.field.
array.expose_ff(azi)
array.expose_ff(ele)
array.expose_ff(sphere)

# Plot normalized far-field cuts in dB:
# - Normalize to isotropic reference Eiso for directivity-like patterns.
# - azi.phi / ele.theta give angular axes, .field.normE contains |E|.
opt.plot_ff(azi.phi, azi.field.normE / opt.Eiso, dB=True)
opt.plot_ff(ele.theta, ele.field.normE / opt.Eiso, dB=True)

# Create a 3D display for the radiation pattern.
display = opt.OptycalDisplay()

# Add a spherical surface plot:
# - 'Etheta' component
# - 'abs' magnitude
# - dB scaling with a floor at -30 dB for visual clarity.
# - rmax the maximum radius to take for the peak gain.
display.add_surf(*sphere.surfplot('Etheta', 'abs', dB=True, dBfloor=-50, rmax=0.5))

display.add(array)
# Render interactive 3D visualization of the array pattern.
display.show()
