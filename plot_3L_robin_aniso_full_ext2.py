#!/usr/bin/env python3
# coding: utf-8

"""plot_3L_robin_aniso_full_ext2.py: Generate figure 3."""

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
filename = 'data_3lay_robin_aniso_full_ext2.txt'

# Medium parameters
rho1 = 2730.0
cp1 = 893.0

k1 = np.array([[150.0, -50.0*np.sqrt(3.0), 50.0],
               [-50.0*np.sqrt(3.0), 250.0, -50.0*np.sqrt(3.0)],
               [50.0, -50.0*np.sqrt(3.0), 300.0]])

rho2 = 1150.0
cp2 = 1700.0

k2 = np.array([[235.0/8.0, -25.0*np.sqrt(3.0)/8.0, 15.0/4.0],
               [-25.0*np.sqrt(3.0)/8.0, 185.0/8.0, -5.0*np.sqrt(3.0)/4.0],
               [15.0/4.0, -5.0*np.sqrt(3.0)/4.0, 75.0/2.0]])

rho3 = 2730.0
cp3 = 893.0

k3 = np.array([[150.0, -50.0*np.sqrt(3.0), -50.0],
               [-50.0*np.sqrt(3.0), 250.0, 50.0*np.sqrt(3.0)],
               [-50.0, 50.0*np.sqrt(3.0), 300.0]])

l1 = 0.03
l2 = 0.005
l3 = 0.025

lg = l1+l2+l3

h1 = 3000.0 / k1[2, 2]
h2 = 4000.0 / k3[2, 2]

# Source strength
Q = 20000.0

# Source beam radius
rw = 0.1

# Transform parameters
Ns = 30
Nsp = 5
Nq = 60
Nphi = 25

# Number of points (time)
Nt = 1000

fig, ax = plt.subplots(1, 1)
markersize = 3


# Source profiles
a = 0.2


def T_ext(s, qx, qy):
    return (s/(s*s-a*a) - 1.0/s)


def profile(q, phi):
    return np.exp(-0.25*q*q*rw*rw)


# Compute solution and plot

num_data = pandas.read_csv(filename, delim_whitespace=True, skiprows=5).values

med = heat_3L_3D_aniso(rho1, cp1, k1, rho2, cp2, k2, rho3, cp3, k3,
                       l1, l2, l3, h1, h2)

med_off = heat_3L_3D_aniso(rho1, cp1, k1, rho2, cp2, k2, rho3, cp3, k3,
                           l1, l2, l3, h1, h2, f_src=None, f_Text=T_ext)

calcfn_f = partial(heat_3L_3D_aniso.calc_laplace, med, z=0, z0=0)
calcfn_m = partial(heat_3L_3D_aniso.calc_laplace, med, z=0.0305, z0=0)
calcfn_b = partial(heat_3L_3D_aniso.calc_laplace, med, z=lg, z0=0)

calcfn_f_off = partial(heat_3L_3D_aniso.calc_laplace, med_off,
                       q=0.0, phiq=0.0, z=0, z0=0)

calcfn_m_off = partial(heat_3L_3D_aniso.calc_laplace, med_off,
                       q=0.0, phiq=0.0, z=0.0305, z0=0)

calcfn_b_off = partial(heat_3L_3D_aniso.calc_laplace, med_off,
                       q=0.0, phiq=0.0, z=lg, z0=0)

t = np.linspace(0, 60, Nt)

start = time.perf_counter()

res_1 = ILT_IFT_brancik(calcfn_f, profile, 0.0, 0.0, t, Ns, Nsp, Nq, Nphi,
                        Q=Q, add_laplace=calcfn_f_off, alpha=a)

stop = time.perf_counter()

print("time: %f" % (stop-start))

res_2 = ILT_IFT_brancik(calcfn_f, profile, 0.06, 0.0, t, Ns, Nsp, Nq, Nphi,
                        Q=Q, add_laplace=calcfn_f_off, alpha=a)

res_3 = ILT_IFT_brancik(calcfn_m, profile, 0.0, 0.0, t, Ns, Nsp, Nq, Nphi,
                        Q=Q, add_laplace=calcfn_m_off, alpha=a)

res_4 = ILT_IFT_brancik(calcfn_m, profile, 0.0, 0.06, t, Ns, Nsp, Nq, Nphi,
                        Q=Q, add_laplace=calcfn_m_off, alpha=a)

res_5 = ILT_IFT_brancik(calcfn_m, profile, 0.0, -0.06, t, Ns, Nsp, Nq, Nphi,
                        Q=Q, add_laplace=calcfn_m_off, alpha=a)

res_6 = ILT_IFT_brancik(calcfn_b, profile, 0.0, 0.0, t, Ns, Nsp, Nq, Nphi,
                        Q=Q, add_laplace=calcfn_b_off, alpha=a)

ax.plot(t, res_1+293.15, 'r', label='$(0,0,0)m$')
ax.plot(num_data[::20, 0], num_data[::20, 1], 'ro', markersize=markersize)

ax.plot(t, res_2+293.15, 'b', label='$(0.06,0,0)m$')
ax.plot(num_data[::20, 0], num_data[::20, 2], 'bo', markersize=markersize)

ax.plot(t, res_3+293.15, 'k', label='$(0,0,0.0305)m$')
ax.plot(num_data[::20, 0], num_data[::20, 11], 'ko', markersize=markersize)

ax.plot(t, res_4+293.15, 'c', label='$(0,0.06,0.0305)m$')
ax.plot(num_data[::20, 0], num_data[::20, 13], 'co', markersize=markersize)

ax.plot(t, res_5+293.15, 'g', label='$(0,-0.06,0.0305)m$')
ax.plot(num_data[::20, 0], num_data[::20, 15], 'go', markersize=markersize)

ax.plot(t, res_6+293.15, 'y', label='$(0,0,0.06)m$')
ax.plot(num_data[::20, 0], num_data[::20, 6], 'yo', markersize=markersize)

plt.xlim(0.0, 35.0)
plt.ylim(270.0, 500.0)

matplotlib.rcParams.update({'font.size': 12})

lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.01), borderaxespad=0)
for legobj in lgd.legendHandles:
    legobj.set_linewidth(2.0)

plt.xlabel('$t / s$')
plt.ylabel('$T / K$')

plt.subplots_adjust(right=0.65)

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.0))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(30.0))

plt.show()
