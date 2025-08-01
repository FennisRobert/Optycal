# Optycal is an open source Python based PO Solver.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.
from typing import List
import numpy as np

from ..space import Edge, Point, Polygon
from shapely import LineString, LinearRing
from numba import njit, f8, i8, typeof
from numba.types import Tuple as TypeTuple
from shapely.validation import explain_validity
import matplotlib.pyplot as plt
from loguru import logger
from itertools import combinations


def is_self_intersecting(loopx, loopy):
    xs = np.array(loopx + loopx[0:2])
    ys = np.array(loopy + loopy[0:2])
    dx = xs[1:]-xs[:-1]
    dy = ys[1:]-ys[:-1]
    cs = dx + 1j*dy
    angs = np.angle(cs[1:]/cs[:-1])
    if abs(np.sum(angs)-2*np.pi) < 1e-3:
        return False
    return True

def plotstate(loopx, loopy, mx, my, tris):
    """
    Function to plot the state of a triangulation.

    Parameters:
    - loopx: x-coordinates of the outer loop of the mesh.
    - loopy: y-coordinates of the outer loop of the mesh.
    - mx: x-coordinates of the triangulation points.
    - my: y-coordinates of the triangulation points.
    - tris: List of tuples representing the triangulation ((1,2,3),(2,31),..) etc.
    """
    
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the outer loop of the mesh with a thick line
    ax.plot(loopx, loopy, 'r-', linewidth=3, label="Outer Loop")

    # Plot the triangulation points as a scatter plot
    ax.scatter(mx, my, c='b', label="Triangulation Points")

    # Plot the triangulation lines
    for tri in tris:
        # Unpack the points that make up each triangle
        
        x = [mx[tri[0]], mx[tri[1]], mx[tri[2]], mx[tri[0]]]
        y = [my[tri[0]], my[tri[1]], my[tri[2]], my[tri[0]]]
        ax.plot(x, y, 'k-', alpha=0.5)  # Thin black lines for the triangles
        if tri==tris[-1]:
            ax.plot(x,y, 'g-', alpha=1.0)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Triangulation Progress')

    # Show the legend
    ax.legend()

    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal', 'box')

    # Display the plot
    plt.show()


class _DebugState:

    def __init__(self, debugmode: bool,
                 meshing_step: int = 0):
        self.debugmode = debugmode
        self.meshing_step = meshing_step
    
    @staticmethod
    def nodebug():
        return _DebugState(False)
    
    @staticmethod
    def debug(step: int):
        return _DebugState(True, step)

_NODEBUG = _DebugState(False)

_FLOATLIST = typeof([1.0, 2.0])
_INTLIST = typeof([1, 2])

@njit(TypeTuple((_FLOATLIST, _FLOATLIST))(_FLOATLIST, _FLOATLIST, _FLOATLIST, _FLOATLIST, _INTLIST), cache=True, nogil=True)
def calc_angles(angles, angles_choice, loopx, loopy, idx):
    N = len(loopx)
    for i in idx:
        i2 = i % N
        i0 = (i2 - 2) % N
        i1 = (i2 - 1) % N
        i3 = (i2 + 1) % N
        i4 = (i2 + 2) % N
        
        dx0 = loopx[i1] - loopx[i0]
        dy0 = loopy[i1] - loopy[i0]
        dx1 = loopx[i1] - loopx[i2]
        dy1 = loopy[i1] - loopy[i2]
        dx2 = loopx[i3] - loopx[i2]
        dy2 = loopy[i3] - loopy[i2]
        dx3 = loopx[i3] - loopx[i4]
        dy3 = loopy[i3] - loopy[i4]
        
        c0 = complex(dx0, dy0)
        c1 = complex(dx1, dy1)
        c2 = complex(dx2, dy2)
        c3 = complex(dx3, dy3)
        
        an1 = (np.angle(c0/c1)) % (2*np.pi)
        angle = (np.angle(c1/c2)) % (2*np.pi)
        an2 = (np.angle(c2/c3)) % (2*np.pi)
        angles[i2] =  angle
        angles_choice[i2] = (0.8*angle + 0.1*(an1 + an2))
    return angles, angles_choice

def calculate_all_angles(angs, angsc, loopx, loopy):
    angs, angsc = calc_angles(angs, angsc, loopx, loopy, list(range(len(loopx))))
    return angs, angsc


def mag(S):
    return np.sqrt(S[0] ** 2 + S[1] ** 2)

def normalize(S):
    return (S[0] / mag(S), S[1] / mag(S))

def rotate(S, angle):
    c = (S[0]+S[1]*1j)*np.exp(1j*angle)
    #c, s = np.cos(angle), np.sin(angle)
    return (c.real, c.imag)

def advancing_front_triangulation(
    polygon: Polygon, dsmax: float, growthrate: float = 0.6, refinement_steps=5, 
    xaxis: np.ndarray = np.array([1,0,0]),
    yaxis: np.ndarray = np.array([0,1,0]), refine=True, debugmode: _DebugState = _NODEBUG, safe_mode: bool = False) -> tuple:
    
    if refine:
        poly = polygon.refine_edges(dsmax)
    else:
        poly = polygon
    tris = []

    loop = poly.vertices[:-1]
    loopx = [p.x for p in loop]
    loopy = [p.y for p in loop]
    
    suma = 0
    for x1, x2, y1, y2 in zip(loopx[:-1], loopx[1:], loopy[:-1], loopy[1:]):
        suma += (x1*y2-x2*y1)

    if suma < 0:
        loopx = loopx[::-1]
        loopy = loopy[::-1]

    L = len(loopx)
    newID = L

    mx = [x for x in loopx]
    my = [y for y in loopy]

    mids = [i for i in range(L)]
    ids = [i for i in range(L)]

    start_loop_ids = [i for i in range(L)]

    angles = [0 for x in loopx]
    angles_choice = [0 for x in loopx]
    angles, angles_choice = calculate_all_angles(angles, angles_choice, loopx, loopy)

    ang1 = np.pi / 2 #* 1.0000001
    ang2 = np.pi * 5 / 6 #* 1.0000001
    
    if min(angles) > np.pi/2:
        angles, angles_choice = calculate_all_angles(angles, angles_choice, loopx, loopy)
        
    counter = {'small': 0, 'medium': 0, 'large1': 0, 'large2': 0}
    increased = 0
    previous_state = len(loopx)
    failure = False
    meshing_step_counter = 0
    

    
    while len(loopx) > 4:
        
        if safe_mode:
            angles, angles_choice = calculate_all_angles(angles, angles_choice, loopx, loopy)
        if len(loopx) > previous_state:
            increased += 1
        elif len(loopx) < previous_state:
            increased = 0
        
        previous_state = len(loopx)

        if increased > 10:
            failure = True
            break
        
        if safe_mode:
            if is_self_intersecting(loopx, loopy):
                plotstate(loopx, loopy, mx, my, tris)
                failure = True
                break
        meshing_step_counter += 1

        if debugmode.debugmode:
            if meshing_step_counter > debugmode.meshing_step - 10:
                plotstate(loopx, loopy, mx, my, tris)

        lid = np.argmin(angles_choice)
        angle = angles[lid]

        N = len(loopx)
        idm2 = (lid - 2) % N
        idm1 = (lid - 1) % N
        idp1 = (lid + 1) % N
        idp2 = (lid + 2) % N

        vidm1 = ids[idm1]
        vid0 = ids[lid]
        vidp1 = ids[idp1]
        

        diagD = np.sqrt((loopx[idm1]-loopx[idp1])**2+(loopy[idm1]-loopy[idp1])**2)
        S2 = (loopx[lid] - loopx[idm1], loopy[lid] - loopy[idm1])
        S3 = (loopx[idp1] - loopx[lid], loopy[idp1] - loopy[lid])
        if (angle <= ang1) and (diagD < 2*dsmax):
            tris.append((vidm1, vid0, vidp1))
            loopx.pop(lid)
            loopy.pop(lid)
            ids.pop(lid)
            angles.pop(lid)
            angles_choice.pop(lid)
            counter['small'] += 1

        elif ang1 < angle <= ang2:
            S1 = (loopx[idm1] - loopx[idm2], loopy[idm1] - loopy[idm2])
            S2 = (loopx[lid] - loopx[idm1], loopy[lid] - loopy[idm1])
            S3 = (loopx[idp1] - loopx[lid], loopy[idp1] - loopy[lid])
            S4 = (loopx[idp2] - loopx[idp1], loopy[idp2] - loopy[idp1])

            D = (
                growthrate * dsmax
                + (1 - growthrate) * (mag(S1) + 2 * mag(S2) + 2 * mag(S3) + mag(S4)) / 6
            )
            cc = np.exp(1j*(np.arctan2(S3[1], S3[0])+angle/2)) 
            vnew = (D * cc.real + loopx[lid], D * cc.imag + loopy[lid])
            inew = newID
            loopx[lid] = vnew[0]
            loopy[lid] = vnew[1]
            ids[lid] = inew

            tris.append((vidm1, vid0, inew))
            tris.append((vidp1, inew, vid0))
            mx.append(vnew[0])
            my.append(vnew[1])
            mids.append(inew)
            counter['medium'] += 1
            newID += 1

        else:# angle > ang2:
            S2 = (loopx[lid] - loopx[idm1], loopy[lid] - loopy[idm1])
            S3 = (loopx[idp1] - loopx[lid], loopy[idp1] - loopy[lid])
            if mag(S2) < mag(S3):
                xy0 = (loopx[idm1], loopy[idm1])
                L = mag(S2)
                newang = np.arccos(L / (2 * min(dsmax, L)))
                dpoint = normalize(rotate(S2, newang))
                L = min(dsmax, L)
                #vnew = (xy0[0] + dpoint[0], xy0[1] + dpoint[1])
                inew = newID
                loopx.insert(lid, xy0[0] + L * dpoint[0])
                loopy.insert(lid, xy0[1] + L * dpoint[1])
                ids.insert(lid, inew)
                angles.insert(lid, 0)
                angles_choice.insert(lid, 0)

                tris.append((vid0, inew, vidm1))
                mx.append(xy0[0] + L*dpoint[0])
                my.append(xy0[1] + L*dpoint[1])
                mids.append(inew)
                counter['large1'] += 1
                newID += 1
            else:
                xy0 = (loopx[lid], loopy[lid])
                L = mag(S3)
                newang = np.arccos(L / (2 * min(dsmax, L)))
                dpoint = normalize(rotate(S3, newang))
                L = min(dsmax, L)
                #vnew = (xy0[0] + dpoint[0], xy0[1] + dpoint[1])
                inew = newID

                loopx.insert(idp1, xy0[0] + L * dpoint[0])
                loopy.insert(idp1, xy0[1] + L * dpoint[1])
                ids.insert(idp1, inew)
                angles.insert(idp1, 0)
                angles_choice.insert(idp1, 0)

                tris.append((vid0, inew, vidp1))
                mx.append(xy0[0] + L*dpoint[0])
                my.append(xy0[1] + L*dpoint[1])
                mids.append(inew)
                counter['large2'] += 1
                newID += 1
        

        angles, angles_choice = calc_angles(angles, angles_choice, loopx, loopy, [idm2, idm1, lid, idp1, idp2])


    if failure is True:
        logger.error('Failure detected')
        if growthrate == 0:
            logger.error(f'Running debug mode with failure at step: {meshing_step_counter}')
            return advancing_front_triangulation(polygon, 
                                      dsmax, 
                                      0, 
                                      refinement_steps=refinement_steps, 
                                      xaxis=xaxis,
                                      yaxis=yaxis, 
                                      refine=refine, 
                                      debugmode = _DebugState(True, meshing_step=meshing_step_counter))
        
        return advancing_front_triangulation(polygon, 
                                      dsmax, 
                                      0, 
                                      refinement_steps=refinement_steps, 
                                      xaxis=xaxis,
                                      yaxis=yaxis, 
                                      refine=refine, 
                                      debugmode = _NODEBUG,
                                      safe_mode=True)
    if len(ids)==4:
        angles, angles_choice = calculate_all_angles(angles, angles_choice, loopx, loopy)
        a1, a2, a3, a4 = angles
        id1, id2, id3, id4 = ids
        if (a2 + a4) <= (a1 + a3):
            tris.append((id1, id2, id3))
            tris.append((id1, id3, id4))
        else:
            tris.append((id1, id2, id4))
            tris.append((id2, id3, id4))
    elif len(ids)==3:
        tris.append((ids[0], ids[1], ids[2]))

    xs = np.array(mx)
    ys = np.array(my)

    dct = {j: i for i, j in enumerate(mids)}
    
    tris2 = tris
    valid_ids = [dct[i] for i in mids if i not in start_loop_ids]
    for qq in range(refinement_steps):
        xs2 = np.zeros(xs.shape)
        ys2 = np.zeros(ys.shape)
        icounter = np.zeros(xs.shape)
        for (I1, I2, I3) in tris2:
            xs2[I1] += (xs[I2] + xs[I3]) / 2
            xs2[I2] += (xs[I1] + xs[I3]) / 2
            xs2[I3] += (xs[I1] + xs[I2]) / 2
            ys2[I1] += (ys[I2] + ys[I3]) / 2
            ys2[I2] += (ys[I1] + ys[I3]) / 2
            ys2[I3] += (ys[I1] + ys[I2]) / 2
            icounter[I1] += 1
            icounter[I2] += 1
            icounter[I3] += 1
        xs[valid_ids] = xs2[valid_ids] / icounter[valid_ids]
        ys[valid_ids] = ys2[valid_ids] / icounter[valid_ids]

    vertices = np.ndarray((3,len(mx)))
    vertices[0,:] = xs*xaxis[0] + ys*yaxis[0]
    vertices[1,:] = xs*xaxis[1] + ys*yaxis[1]
    vertices[2,:] = xs*xaxis[2] + ys*yaxis[2]
    return vertices, np.array(tris2).T, start_loop_ids

def advancing_front_triangulation_ps(
    t1: list, t2: list, ps, growthrate: float = 0.6, refinement_steps=5,  ) -> tuple:
    polygon = Polygon([Point(x, y) for x, y in zip(t1, t2)])
    return advancing_front_triangulation(polygon, ps, growthrate, refinement_steps)

# def advancing_front_mesh(polygon: Polygon, dsmax: float, growthrate: float = 0.6, refinement_steps=5, 
#     xaxis: np.ndarray = np.array([1,0,0]),
#     yaxis: np.ndarray = np.array([0,1,0])):
#     from emerge.threed.mesh import Mesh
#     vertices, tris, boundary = advancing_front_triangulation(polygon, dsmax, growthrate, refinement_steps, xaxis, yaxis)
#     #vertices = [Point(x, y) for x, y in zip(mx, my)]
#     mesh = Mesh(vertices)
#     mesh.set_triangles(tris)
#     mesh._fill_complete()
#     mesh.boundary_vertices = boundary

#     return mesh
