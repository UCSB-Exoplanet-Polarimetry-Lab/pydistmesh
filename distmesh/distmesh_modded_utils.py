!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy.spatial as spspatial
from scipy.interpolate import interp1d

# Local imports
import distmesh.mlcompat as ml
import distmesh.utils as dmutils
import pickle
from matplotlib import pyplot as plt
from IPython import get_ipython

from scipy.spatial import Delaunay

from mpi4py import MPI

from _delaunay_class import DelaunayTriangulation as DT2

import time

import copy
import pandas as pd

import distmesh as dm
from SeismicMesh import geometry
from SeismicMesh.generation import utils as mutils

from scipy.interpolate import RegularGridInterpolator

from matplotlib.tri import Triangulation

def getRings(data,center,R1,R2):
    imin = center[0] - R2
    imax = center[0] + R2 + 1
    jmin = center[1] - R2
    jmax = center[1] + R2 + 1

    target = []
    X=np.linspace(-0.5,0.5,imax-imin)
    Y=np.linspace(-0.5,0.5,jmax-jmin)
    Tht = []
    for i in np.arange(imin, imax):
        for j in np.arange(jmin, jmax):
            ij = np.array([i,j])
            dist = np.linalg.norm(ij - np.array(center))
            if dist > R1 and dist <= R2:
                target.append([i,j,data[i][j]])
                Tht.append(np.arctan2(X[i-imin],Y[j-jmin])%(2*np.pi))
    target = np.array(target)
    Tht = np.array(Tht)
    return target,Tht

def Extract(lst,idx):
    return [item[idx] for item in lst]

def _compute_forces(p, t, fh, h0):#, L0mult):
#     Fscale=1.2;
    dim=2
    L0mult = 1 + 0.4 / 2 ** (dim - 1)
    """Compute the forces on each edge based on the sizing function"""
    dim = p.shape[1]
    N = p.shape[0]
    edges = _get_edges(t)
    barvec = p[edges[:, 0]] - p[edges[:, 1]]  # List of bar vectors
    L = np.sqrt((barvec**2).sum(1))  # L = Bar lengths
    L[L == 0] = np.finfo(float).eps
    hedges = fh(p[edges].sum(1) / 2)
    L0 = hedges * L0mult * ((L**dim).sum() / (hedges**dim).sum()) ** (1.0 / dim)
    del(hedges)
    F = L0 - L
    F[F < 0] = 0  # Bar forces (scalars)
    Fvec = (
        F[:, None] / L[:, None].dot(np.ones((1, dim))) * barvec
    )  # Bar forces (x,y components)
    
    del(L)
    del(barvec)
    Flen = len(F)
    del(F)
    
    Ftot = mutils.dense(
        edges[:, [0] * dim + [1] * dim],
        np.repeat([list(range(dim)) * 2], Flen, axis=0),
        np.hstack((Fvec, -Fvec)),
        shape=(N, dim),
    )
    return Ftot

def unique_rows_fast(A, return_index=False, return_inverse=False):

    A = np.require(A, requirements='C')

    A = pd.DataFrame(A)
    B = np.array(A.drop_duplicates().sort_values(0))
    
    return B

#### Custom faster 2d interpolation function

def interp2d_fast(x, y, xp, yp, zp):
    """
    Bilinearly interpolate over regular 2D grid.

    `xp` and `yp` are 1D arrays defining grid coordinates of sizes :math:`N_x`
    and :math:`N_y` respectively, and `zp` is the 2D array, shape
    :math:`(N_x, N_y)`, containing the gridded data points which are being
    interpolated from. Note that the coordinate grid should be regular, i.e.
    uniform grid spacing. `x` and `y` are either scalars or 1D arrays giving
    the coordinates of the points at which to interpolate. If these are outside
    the boundaries of the coordinate grid, the resulting interpolated values
    are evaluated at the boundary.

    Parameters
    ----------
    x : 1D array or scalar
        x-coordinates of interpolating point(s).
    y : 1D array or scalar
        y-coordinates of interpolating point(s).
    xp : 1D array, shape M
        x-coordinates of data points zp. Note that this should be a *regular*
        grid, i.e. uniform spacing.
    yp : 1D array, shape N
        y-coordinates of data points zp. Note that this should be a *regular*
        grid, i.e. uniform spacing.
    zp : 2D array, shape (M, N)
        Data points on grid from which to interpolate.

    Returns
    -------
    z : 1D array or scalar
        Interpolated values at given point(s).

    """
    # if scalar, turn into array
    scalar = False
    if not isinstance(x, (list, np.ndarray)):
        scalar = True
        x = np.array([x])
        y = np.array([y])

    # grid spacings and sizes
    hx = xp[1] - xp[0]
    hy = yp[1] - yp[0]
    Nx = xp.size
    Ny = yp.size

    # snap beyond-boundary points to boundary
    x[x < xp[0]] = xp[0]
    y[y < yp[0]] = yp[0]
    x[x > xp[-1]] = xp[-1]
    y[y > yp[-1]] = yp[-1]

    # find indices of surrounding points
    i1 = np.floor((x - xp[0]) / hx).astype(int)
    i1[i1 == Nx - 1] = Nx - 2
    j1 = np.floor((y - yp[0]) / hy).astype(int)
    j1[j1 == Ny - 1] = Ny - 2
    i2 = i1 + 1
    j2 = j1 + 1

    # get coords and func at surrounding points
    x1 = xp[i1]
    x2 = xp[i2]
    y1 = yp[j1]
    y2 = yp[j2]
    z11 = zp[i1, j1]
    z21 = zp[i2, j1]
    z12 = zp[i1, j2]
    z22 = zp[i2, j2]

    # interpolate
    #def inter(x,y):
    t11 = z11 * (x2 - x) * (y2 - y)
    t21 = z21 * (x - x1) * (y2 - y)
    t12 = z12 * (x2 - x) * (y - y1)
    t22 = z22 * (x - x1) * (y - y1)
    z = (t11 + t21 + t12 + t22) / (hx * hy)
    if scalar:
        z = z[0]
    return z

def azimuthal_interp(datamap):
    mapcenter = (int(datamap.shape[0]/2),int(datamap.shape[1]/2))
    Ringmap,thtmap = getRings(datamap,mapcenter,int(int(datamap.shape[0]/2)/2),int(int(datamap.shape[0]/2)/2)+1)
    rm = Extract(Ringmap,2)
    interp_1d_func = interp1d(np.linspace(0,2*np.pi,len(rm),np.array(rm)[np.argsort(thtmap)][::-1])
    return interp_1d_func

def _get_topology(dt):
    """Get points and entities from :clas:`CGAL:DelaunayTriangulation2/3` object"""
    p = dt.get_finite_vertices()
    t = dt.get_finite_cells()
    return p, t
    
def _get_edges(t):
    """Describe each bar by a unique pair of nodes"""
    dim = t.shape[1] - 1
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return geometry.unique_edges(edges)
