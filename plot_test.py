import optycal as opt
import numpy as np
import matplotlib.pyplot as plt

def dB(x):
    return 20*np.log10(np.abs(x))

array = opt.AntennaArray(1e9)
array.add_2d_array(opt.taper(5), opt.taper(5), [0, 0.15, 0], [0, 0, 0.15], 0)

azi, ele = opt.FF1D.aziele(1)

ffazi = array.expose_ff(azi)
ffele = array.expose_ff(ele)

sphmsh = opt.generate_sphere(np.array([0,0,0]), 1, 0.1, opt.GCS)
sphere = opt.Surface(sphmsh, opt.FRES_AIR)

rect = opt.generate_rectangle(1, 1, [0, 1, 0], [0,0,1], 0.09, opt.GCS, origin=(3,0,0))
rect_surf = opt.Surface(rect, opt.FRES_PEC)

array.expose_surface(sphere)
sphere.expose_surface(rect_surf)
rect_surf.expose_surface(sphere)

ffazis = sphere.expose_ff(azi)
ffeles = sphere.expose_ff(ele)


display = opt.OptycalDisplay()

display.add(array)
display.add(sphere, opacity=0.5, field='Ez', quantity='real')
display.add(rect_surf, opacity=0.5, field='Ez', quantity='real')

display.show()