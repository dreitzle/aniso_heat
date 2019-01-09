#!/usr/bin/env python3
# coding: utf-8

"""plot_3L_robin_aniso_full_zcut.py: Generate figure 6."""

import numpy as np
from functools import partial
import pandas

from numeric_transform import ILT_IFT_hyperbolic

from heatNL import heat_layer
from heatNL import heat_NL_3D_aniso

import matplotlib.pyplot as plt

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2018, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

# Numeric reference data
filename = 'data_Nlay_robin_aniso_full_zcut_R.txt'

# Medium parameters

rho_alu = 2730.0
cp_alu = 893.0

rho_nylon = 1150.0
cp_nylon = 1700.0

k1 = np.array([[150.0, -50.0*np.sqrt(3.0), 50.0],
               [-50.0*np.sqrt(3.0), 250.0, -50.0*np.sqrt(3)],
               [50.0, -50.0*np.sqrt(3), 300.0]])

l1 = 0.01

layer1 = heat_layer(rho_alu, cp_alu, k1, l1)


k2 = np.array([[235.0/2.0, -25.0*np.sqrt(3.0)/2.0, 15.0/1.0],
               [-25.0*np.sqrt(3.0)/2.0, 185.0/2.0, -5.0*np.sqrt(3.0)/1.0],
               [15.0/1.0, -5.0*np.sqrt(3.0)/1.0, 75.0*2.0]])

l2 = 0.005

layer2 = heat_layer(rho_nylon, cp_nylon, k2, l2)


k3 = np.array([[150.0, -50.0*np.sqrt(3.0), -50.0],
               [-50.0*np.sqrt(3.0), 250.0, 50.0*np.sqrt(3.0)],
               [-50.0, 50.0*np.sqrt(3.0), 300.0]])

l3 = 0.015

layer3 = heat_layer(rho_alu, cp_alu, k3, l3)


kr = 2e-4*np.array([[1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0]])

lr = 1e-8  # Contact layer between layers 3 and 4

layerR = heat_layer(rho_alu, cp_alu, kr, lr)


k4 = np.array([[675.0/4.0, -75.0*np.sqrt(3.0)/4.0, -25.0*np.sqrt(3.0)/2.0],
               [-75.0*np.sqrt(3.0)/4.0, 825.0/4.0, 75.0/2.0],
               [-25.0*np.sqrt(3.0)/2.0, 75.0/2.0, 175.0]])

l4 = 0.008

layer4 = heat_layer(rho_alu, cp_alu, k4, l4)


k5 = np.array([[155.0, 0.0, 0.0],
               [0.0, 155.0, 0.0],
               [0.0, 0.0, 155.0]])

l5 = 0.012

layer5 = heat_layer(rho_alu, cp_alu, k5, l5)


k6 = np.array([[425.0, -25.0, 25.0*np.sqrt(6.0)],
               [-25.0, 425.0, -25.0*np.sqrt(6.0)],
               [25.0*np.sqrt(6.0), -25.0*np.sqrt(6.0), 350.0]])

l6 = 0.02

layer6 = heat_layer(rho_alu, cp_alu, k6, l6)


k7 = np.array([[250.0, 0.0, 0.0],
               [0.0, 150.0, 0.0],
               [0.0, 0.0, 300.0]])

l7 = 0.01

layer7 = heat_layer(rho_alu, cp_alu, k7, l7)


layers = [layer1, layer2, layer3, layerR, layer4, layer5, layer6, layer7]

lg = 0.0
for layer in layers:
    lg += layer.lz


h1 = 3000.0 / k1[2, 2]
h2 = 4000.0 / k7[2, 2]

# Source strength
Q = 20000.0

# Source beam radius
rw = 0.1

# Transform parameters
Ns = 15
Nq = 50
Nphi = 25
x = 0
y = 0

# Transform times
t1 = 3.0
t2 = 6.0
t3 = 12.0


# Source profile
def profile(q, phi):
    return np.exp(-0.25*q*q*rw*rw)


# Compute solution and plot

num_data = pandas.read_csv(filename, delim_whitespace=True, skiprows=7).values

med = heat_NL_3D_aniso(layers, h1, h2)

z = np.linspace(0, lg, 200)

res1 = np.empty_like(z)
res2 = np.empty_like(z)
res3 = np.empty_like(z)

for index, zp in enumerate(z):
    calcfn = partial(heat_NL_3D_aniso.calc_laplace, med, z=zp, z0=0.0)
    res1[index] = Q*ILT_IFT_hyperbolic(calcfn, profile, 0.0, 0.0, t1, Ns, Nq, Nphi)
    res2[index] = Q*ILT_IFT_hyperbolic(calcfn, profile, 0.0, 0.0, t2, Ns, Nq, Nphi)
    res3[index] = Q*ILT_IFT_hyperbolic(calcfn, profile, 0.0, 0.0, t3, Ns, Nq, Nphi)

markersize = 3
fig, ax = plt.subplots()

ax.plot(z*1e3, res1, 'b', label='$t = 3$s')
ax.plot(z*1e3, res2, 'g', label='$t = 6$s')
ax.plot(z*1e3, res3, 'r', label='$t = 12$s')

ax.plot(num_data[::5, 0], num_data[::5, 2]-293.15, 'bo', markersize=markersize)
ax.plot(num_data[::5, 0], num_data[::5, 3]-293.15, 'go', markersize=markersize)
ax.plot(num_data[::5, 0], num_data[::5, 5]-293.15, 'ro', markersize=markersize)

ax.axvline((l1)*1e3, linestyle='dashed', color='r')
ax.axvline((l1+l2)*1e3, linestyle='dashed', color='r')
ax.axvline((l1+l2+l3)*1e3, linestyle='dashed', color='r')
ax.axvline((l1+l2+l3+l4)*1e3, linestyle='dashed', color='r')
ax.axvline((l1+l2+l3+l4+l5)*1e3, linestyle='dashed', color='r')
ax.axvline((l1+l2+l3+l4+l5+l6)*1e3, linestyle='dashed', color='r')

ax.set_xlim(0, lg*1e3)
ax.set_ylim(0, None)

lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.01), borderaxespad=0)
for legobj in lgd.legendHandles:
    legobj.set_linewidth(2.0)

plt.xlabel('$z / $mm')
plt.ylabel('$\Delta T / $K')

plt.show()
