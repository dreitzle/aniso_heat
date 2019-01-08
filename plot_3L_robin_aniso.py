# coding: utf-8

"""plot_3L_robin_aniso.py: Generate figure 2."""

import numpy as np
from functools import partial
import pandas

from numeric_transform import ILT_IFT_brancik

from heat3L import heat_3L_3D_aniso

import matplotlib.pyplot as plt
import matplotlib as matplotlib

import time

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2018, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

# Numeric reference data
filename = 'data_3lay_robin_aniso.txt'

# Medium parameters
rho1 = 2730.0
cp1 = 893.0
k1 = np.array([[200.0, 0.0, 0.0],
               [0.0, 400.0, 0.0],
               [0.0, 0.0, 155.0]])

rho2 = 1150.0
cp2 = 1700.0
k2 = np.array([[20.0, 0.0, 0.0],
               [0.0, 20.0, 0.0],
               [0.0, 0.0, 20.0]])

rho3 = 2730.0
cp3 = 893.0
k3 = np.array([[400.0, 0.0, 0.0],
               [0.0, 200.0, 0.0],
               [0.0, 0.0, 155.0]])

l1 = 0.03
l2 = 0.005
l3 = 0.025
h1 = 3000.0 / k1[2, 2]
h2 = 4000.0 / k3[2, 2]

lg = l1+l2+l3

# Source strength
Q = 20000

# Source beam radius
rw = 0.1

# Transform parameters
Ns = 30
Ns2 = 5
Nq = 60
Nphi = 25

# Number of points (time)
Nt = 1000


# Source profile
def profile(q, phi):
    return np.exp(-0.25*q*q*rw*rw)


# Compute solution and plot

fig, ax = plt.subplots(1, 1)
markersize = 3

med = heat_3L_3D_aniso(rho1, cp1, k1, rho2, cp2, k2, rho3, cp3, k3,
                       l1, l2, l3, h1, h2)

calcfn_b = partial(heat_3L_3D_aniso.calc_laplace, med, z=0.0, z0=0.0)

x = 0.0
y = 0.0

num_data = pandas.read_csv(filename, delim_whitespace=True, skiprows=5).as_matrix()

t = np.linspace(0, 60, Nt)

start = time.perf_counter()
res = Q*ILT_IFT_brancik(calcfn_b, profile, x, y, t, Ns, Ns2, Nq, Nphi)
stop = time.perf_counter()

print("time: %f" % (stop-start))

ax.plot(t, res+293.15, 'c', label='$(0,0,0)m$')
ax.plot(num_data[::20, 0], num_data[::20, 1], 'co', markersize=markersize)

##

calcfn_b = partial(heat_3L_3D_aniso.calc_laplace, med, z=0.06, z0=0.0)

x = 0.0
y = 0.0

res = Q*ILT_IFT_brancik(calcfn_b, profile, x, y, t, Ns, Ns2, Nq, Nphi)

ax.plot(t, res+293.15, 'r', label='$(0,0,0.06)m$')
ax.plot(num_data[::20, 0], num_data[::20, 11], 'ro', markersize=markersize)


###

calcfn_b = partial(heat_3L_3D_aniso.calc_laplace, med, z=0.0305, z0=0.0)

x = 0.0
y = 0.0

res = Q*ILT_IFT_brancik(calcfn_b, profile, x, y, t, Ns, Ns2, Nq, Nphi)

ax.plot(t, res+293.15, 'k', label='$(0,0,0.0305)m$')
ax.plot(num_data[::20, 0], num_data[::20, 6], 'ko', markersize=markersize)

###

calcfn_b = partial(heat_3L_3D_aniso.calc_laplace, med, z=0.0305, z0=0.0)

x = 0.0
y = 0.06

res = Q*ILT_IFT_brancik(calcfn_b, profile, x, y, t, Ns, Ns2, Nq, Nphi)

ax.plot(t, res+293.15, 'm', label='$(0.0,0.06,0.0305)m$')
ax.plot(num_data[::20, 0], num_data[::20, 8], 'mo', markersize=markersize)

###

calcfn_b = partial(heat_3L_3D_aniso.calc_laplace, med, z=0.0305, z0=0.0)

x = 0.06
y = 0.0

res = Q*ILT_IFT_brancik(calcfn_b, profile, x, y, t, Ns, Ns2, Nq, Nphi)

ax.plot(t, res+293.15, 'b', label='$(0.06,0.0,0.0305)m$')
ax.plot(num_data[::20, 0], num_data[::20, 7], 'bo', markersize=markersize)

###

matplotlib.rcParams.update({'font.size': 12})

lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.01), borderaxespad=0)
for legobj in lgd.legendHandles:
    legobj.set_linewidth(2.0)

plt.xlabel('$t / s$')
plt.ylabel('$T / K$')

plt.subplots_adjust(right=0.65)

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.0))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20.0))

plt.show()
