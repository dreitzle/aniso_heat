#!/usr/bin/env python3
# coding: utf-8

"""plot_3L_robin_aniso_full_moving_FFT.py: Generate figures 4 and 5."""

import numpy as np
from functools import partial
import pandas
import bz2
import io

from numeric_transform import ILT_FFT_hyperbolic

from heat3L import heat_3L_3D_aniso

import matplotlib.pyplot as plt

import time

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2018, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

# Numeric reference data
filename = 'data_3lay_robin_aniso_full_moving_grid.txt.bz2'

# Medium parameters

ux = 0.01
uy = 0.0

rho1 = 2730.0
cp1 = 893.0
k1 = np.array([[150.0, -50.0*np.sqrt(3.0), 50.0],
               [-50.0*np.sqrt(3.0), 250.0, -50.0*np.sqrt(3.0)],
               [50.0, -50.0*np.sqrt(3.0), 300.0]])

l1 = 0.03

rho2 = 1150.0
cp2 = 1700.0

k2 = np.array([[235.0/8.0, -25.0*np.sqrt(3.0)/8.0, 15.0/4.0],
               [-25.0*np.sqrt(3.0)/8.0, 185.0/8.0, -5.0*np.sqrt(3.0)/4.0],
               [15.0/4.0, -5.0*np.sqrt(3.0)/4.0, 75.0/2.0]])

l2 = 0.005


rho3 = 2730.0
cp3 = 893.0

k3 = np.array([[150.0, -50.0*np.sqrt(3.0), -50.0],
               [-50.0*np.sqrt(3.0), 250.0, 50.0*np.sqrt(3.0)],
               [-50.0, 50.0*np.sqrt(3.0), 300.0]])

l3 = 0.025

lg = l1+l2+l3

h1 = 3000.0 / k1[2, 2]
h2 = 4000.0 / k3[2, 2]

# Source strength
Q = 20000.0

# Source beam radius
rw = 0.1

# Transform parameters
Ns = 15
NFFT = 501
xFFT = 1.0
dFFT = xFFT/NFFT


# Source profile
def profilexy(qx, qy):
    return np.exp(-0.25*(qx*qx+qy*qy)*rw*rw)


# Compute solution and plot

fft_data_x = 1e3*np.linspace(-xFFT/2.0, xFFT/2.0, NFFT)

# Compressed file, decompress first
datafile = io.StringIO(bz2.BZ2File(filename).read().decode('us-ascii'))
num_data = pandas.read_csv(datafile, delim_whitespace=True, skiprows=8).as_matrix()

num_data_x = num_data[:, 0]
num_data_y = num_data[:, 1]

t1 = 10
num_data_t1 = num_data[:, 4]

t2 = 60
num_data_t2 = num_data[:, 9]

del num_data

num_data_x = num_data_x.reshape((1001, 1001))
num_data_y = num_data_y.reshape((1001, 1001))
num_data_t1 = num_data_t1.reshape((1001, 1001))
num_data_t2 = num_data_t2.reshape((1001, 1001))

med = heat_3L_3D_aniso(rho1, cp1, k1, rho2, cp2, k2, rho3, cp3, k3,
                       l1, l2, l3, h1, h2, ux, uy)

calcfn_f = partial(heat_3L_3D_aniso.calc_laplace_ft, med, z=0.0, z0=0.0)

start = time.perf_counter()
res_1 = 293.15 + Q*ILT_FFT_hyperbolic(calcfn_f, profilexy, NFFT, dFFT, t1, Ns)
stop = time.perf_counter()
print("time: %f" % (stop-start))

res_2 = 293.15 + Q*ILT_FFT_hyperbolic(calcfn_f, profilexy, NFFT, dFFT, t2, Ns)

fig1, ax1 = plt.subplots()

im1 = ax1.contourf(num_data_x, num_data_y, num_data_t1, 20, vmin=293.15)

vmin, vmax = im1.get_clim()
cax = fig1.add_axes([0.9, 0.111, 0.02, 0.767])
ax1.set_xlim(-200.0, 500.0)
ax1.set_ylim(-200.0, 200.0)
cbar = plt.colorbar(im1, cax=cax)

fig2, ax2 = plt.subplots()

im2 = ax2.contourf(fft_data_x, fft_data_x, np.transpose(res_1),
                   20, vmin=vmin, vmax=vmax)

ax2.set_xlim(-200.0, 500.0)
ax2.set_ylim(-200.0, 200.0)

fig3, ax3 = plt.subplots()

im3 = ax3.contourf(num_data_x, num_data_y, num_data_t2, 20, vmin=293.15)

vmin, vmax = im3.get_clim()
cax = fig3.add_axes([0.9, 0.111, 0.02, 0.767])
ax3.set_xlim(-200.0, 500.0)
ax3.set_ylim(-200.0, 200.0)
cbar = plt.colorbar(im3, cax=cax)

fig4, ax4 = plt.subplots()

im4 = ax4.contourf(fft_data_x, fft_data_x, np.transpose(res_2),
                   20, vmin=vmin, vmax=vmax)

ax4.set_xlim(-200.0, 500.0)
ax4.set_ylim(-200.0, 200.0)

plt.show()
