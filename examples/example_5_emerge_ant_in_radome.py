import emerge as em
import optycal as opt
from emerge.plot import plot_sp, plot_ff_polar, plot_ff
import numpy as np

"""This example demonstrates how to simulate an antenna element inside a radome environemnt

It requires the emerge FEM package to run.

This demo will illustrate how EMerge and Optycal can be used together.

Because of how Optycal and EMerge work, the simulation will not combine the two simulations in a bi direcitional way.
The FEM simulation will not know about the radome and reflections wil not be taken into account in the S11.

The simulation is intended to show the deformation of the beam pattern and potential internal reflections inside the radome.

We will simulate a simple monopole over a disc ground plane at 2.45 GHz, enclosed in a hemispherical radome made of a dielectric material.
"""


mm = 0.001

f0 = 2.45e9  # Operating frequency [Hz]
c0 = 299792458  # Speed of light in vacuum [m/s]
wavelength = c0 / f0  # Wavelength corresponding to the operating frequency
k0 = 2 * em.lib.PI / wavelength  # Wavenumber corresponding to the operating frequency
ant_L = 0.21 * wavelength  # Length of monopole
gap = 2*mm                 # Gap for the feed.
ant_R = 2*mm
ground_R = 1.5 * wavelength  # Radius of ground plane
radome_R = 3 * wavelength  # Radius of radome

## First we start with the monopole

sim = em.Simulation('Monopole')

ant = em.geo.Cylinder(ant_R, ant_L, em.GCS.displace(0,0,gap), Nsections=12).set_material(em.lib.COPPER)
feed = em.geo.Cylinder(ant_R, gap, em.GCS.displace(0,0,0))
disc = em.geo.Disc((0,0,0), ground_R, (0,0,1)).set_material(em.lib.PEC)
air = em.geo.HalfSphere(ground_R, (0,0,0), (0,0,1)).background()

sim.commit_geometry()
sim.mw.set_frequency(f0)


sim.mw.bc.LumpedPort(feed.shell, 1, 2*em.lib.PI*ant_R, gap, em.ZAX)
sim.mw.bc.AbsorbingBoundary(air.outside)

sim.mesher.set_boundary_size(ant, ant_R)
sim.mesher.set_boundary_size(air, wavelength/5)
sim.generate_mesh()
sim.view(plot_mesh=True)

data = sim.mw.run_sweep()

g = data.scalar.grid
field = data.field[0]
print(f'Antenna S11 = {20*np.log10(np.abs(g.S(1,1)))}')

sim.display.add_objects(*sim.all_geos())
sim.display.cbar('Antenna E-field', clim=(0,1000)).add_surf(*field.cutplane(wavelength/25, y=0).scalar('normE'))
sim.display.show()

farfield = field.farfield_2d(em.ZAX, em.YAX, air.outside, (-90, 90))

plot_ff_polar(farfield.ang, farfield.normE/em.lib.EISO)

## Now lets integrate it into an optycal simulation

# We import the surface of the radiation into our Optycal simulation
ant_surf = opt.Surface.import_model(*field.optycal_surface(air.outside))
monopole_surf = opt.Surface.import_model(*field.optycal_surface(ant.boundary()))
monopole_surf.fresnel = opt.FRES_PEC
disc_surf = opt.Surface.import_model(*field.optycal_surface(disc))
disc_surf.fresnel = opt.FRES_PEC

# We create the radome surface materials
laminate = opt.Material(er=3.5, tand=0.05, color="#c6c6c6")
foam = opt.lib.FOAM_AIREX_C70
# Next we create a stack for the radome based on a laminate and foam sandwich. 
radome_material_stack = opt.MultiLayer(ant_surf.k0, [laminate, foam, laminate],[2*mm, 12*mm, 2*mm])

# We generate the radome mesh as a sphere with apartial theta range
radome_mesh = opt.generate_sphere(np.array([0,0,0]), radome_R, wavelength/5, opt.GCS, thrange=(0,np.pi/2))
# We combine the radome mesh and material stack into a surface
radome_surf = opt.Surface(radome_mesh, radome_material_stack)

# For plotting we create a azimuthal and elevation far field cuts
ffazi, ffele = opt.FF1D.aziele()
ff3d = opt.FF2D.halfsphere()

# Next we expose the different "objects" by our antenna and radome surface
ant_surf.expose_surface(radome_surf, 2)
ff_azi_ant = ant_surf.expose_ff(ffazi, 2)
ff_ele_ant = ant_surf.expose_ff(ffele, 2)
ff_azi_radome = radome_surf.expose_ff(ffazi, 1)
ff_ele_radome = radome_surf.expose_ff(ffele, 1)
field_radome = radome_surf.expose_ff(ff3d, 1)

# Finally we plot the results
plot_ff_polar(ffazi.phi, [ff_azi_ant.normE/em.lib.EISO, ff_azi_radome.normE/em.lib.EISO], labels=['Antenna only','With radome'])
plot_ff_polar(ffele.theta, [ff_ele_ant.normE/em.lib.EISO, ff_ele_radome.normE/em.lib.EISO], labels=['Antenna only','With radome'])


# And a 3D plot
disp = opt.OptycalDisplay()
#disp.add_surface_object(ant_surf, field='Ez', quantity='real')
disp.add_surface_object(radome_surf, field='normE', opacity=0.3)
disp.add_surface_object(monopole_surf)
disp.add_surface_object(disc_surf)
disp.add_surf(*ff3d.surfplot('normE','abs', rmax=radome_R*0.6, offset=(0,0,radome_R*1.1)))
disp.show()