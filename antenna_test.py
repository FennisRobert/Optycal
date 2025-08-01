import optycal as opt
import numpy as np
import matplotlib.pyplot as plt

def dB(x):
    return 20*np.log10(np.abs(x))

ant = opt.Antenna(0,0,0, 1e9, opt.GCS)

azi, ele = opt.FF1D.aziele(1)

ffazi = ant.expose_ff(azi)
ffele = ant.expose_ff(ele)

fig, ax = plt.subplots()

ax.plot(azi.phi*180/np.pi, dB(ffazi.normE), label='Azimuthal')
ax.plot(ele.theta*180/np.pi, dB(ffele.normE), label='Azimuthal')

plt.show()

sphmsh = opt.generate_sphere(np.array([0,0,0]), 1, 0.1, opt.GCS)
sphere = opt.Surface(sphmsh, opt.FRES_AIR, 2)

ant.expose_surface(sphere)

ffazis = sphere.expose_ff(azi)
ffeles = sphere.expose_ff(ele)

fig, ax = plt.subplots()

ax.plot(azi.phi*180/np.pi, dB(ffazi.normE), label='Azimuthal')
ax.plot(ele.theta*180/np.pi, dB(ffele.normE), label='Azimuthal')
ax.plot(azi.phi*180/np.pi, dB(ffazis.normE), label='Azimuthal')
ax.plot(ele.theta*180/np.pi, dB(ffeles.normE), label='Azimuthal')

plt.show()

display = opt.OptycalDisplay()

display.add(sphere, field='Ez', quantity='real')

display.show()