import optycal as opt

"""Example 1: Simple Single antenna simulation

In this example we will look at the very basics of Optycal. What it can do with simple antennas. 
In this example we will try just one.

For EMerge users: Contrary to EMerge, you don't need a Simulation object in Optycal. You can just start working.
"""

# Frequency definition
freq = 2e9 # 2Ghz

# The simplest antenna in Optycal is just an antenna object. We define its coordinate, frequency and two antenna patterns.
# In Optycal we use two functions for this, a near-field and far-field function. the difference is that Near-Field functions
# are a function of (x,y,z) while farfield functions are a function of (θ,Φ).

ant = opt.Antenna(0,0,0, freq, opt.GCS, opt.dipole_pattern_nf, opt.dipole_pattern_ff)

# Next we use the Farfield-1D (FF1D) class to simply create a farfield data structure filled with coordinates to
# compute the E-field at. Think of these as containers to immediately plug in all relevant data.

azi, ele = opt.FF1D.aziele(dangle=1)

# We can also use a 2D farfield dataset in a sphere. Notice that 1D and 2D refers to the structure of the data itself, 1D vs 2D arrays.
sphere = opt.FF2D.sphere(1)

# The anntenna and other class objects in Optycal have an expose_... method. In this case we expose our Farfield datasets with the expose_ff.
# The data gets immediately stored in the dataset.

ant.expose_ff(azi)
ant.expose_ff(ele)
ant.expose_ff(sphere)

# Now we want to plot our results. For simplicity we have plot-ff functions in Optycal.
# As you can see, we have the phi, theta, and field data at our disposal. The solutions are in .field. In this case we plot normE.
opt.plot_ff(azi.phi, azi.field.normE)
opt.plot_ff(ele.theta, ele.field.normE)

# We can also display our antenna diagram. We first create a PyVista based display object.
display = opt.OptycalDisplay()

# then we just add our surface. To quickly create the data we call the surfplot method on the field solution
display.add_surf(*sphere.surfplot('Etheta', ))

# And we show.
display.show()