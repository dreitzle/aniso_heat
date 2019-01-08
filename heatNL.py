# coding: utf-8

"""heatNL.py: N-layered solution implementation"""

import numpy as np
import scipy.linalg as la
import collections

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2018, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"


class heat_layer:
    """Class representing a single layer."""

    def __init__(self, rho, cp, k, lz, ux=0.0, uy=0.0):
        """
        :param rho: Mass density.
        :param cp: Heat capacity.
        :param k: Thermal conductivity (3x3 symmetric Matrix).
        :param lz: Layer thickness.
        :param ux: x-component of layer velocity.
        :param uy: y-component of layer velocity.
        """
        assert isinstance(k, np.ndarray)
        assert k.shape == (3, 3)
        assert np.allclose(k, k.T, atol=1e-8)

        self.rho = rho
        self.cp = cp
        self.k = k
        self.lz = lz

        self.ux = ux
        self.uy = uy


# default source functions

def _src_switch(s):
    return 1.0/s


def _Text_null(s, qx, qy):
    return 0.0


# solution implementation

class heat_NL_3D_aniso:
    """N-layered solution implementation."""

    def __init__(self, geometry, h1=0, h2=0):
        """
        :param geometry: List of heat_layer, top to bottom.
        :param h1: Heat transfer coefficient of upper boundary.
        :param h2: Heat transfer coefficient of lower boundary.
        """

        self.h1 = h1
        self.h2 = h2

        self.Nl = len(geometry)

        self.lg = 0.0

        self.rho = np.empty(self.Nl, dtype=float)
        self.cp = np.empty(self.Nl, dtype=float)

        self.kx = np.empty(self.Nl, dtype=float)
        self.ky = np.empty(self.Nl, dtype=float)
        self.kz = np.empty(self.Nl, dtype=float)
        self.kxy = np.empty(self.Nl, dtype=float)
        self.kxz = np.empty(self.Nl, dtype=float)
        self.kyz = np.empty(self.Nl, dtype=float)

        self.ll = np.empty(self.Nl, dtype=float)

        self.ux = np.empty(self.Nl, dtype=float)
        self.uy = np.empty(self.Nl, dtype=float)

        for idx, layer in enumerate(geometry):

            self.rho[idx] = layer.rho
            self.cp[idx] = layer.cp
            self.ll[idx] = layer.lz
            self.lg += layer.lz

            self.kx[idx] = layer.k[0, 0]
            self.ky[idx] = layer.k[1, 1]
            self.kz[idx] = layer.k[2, 2]
            self.kxy[idx] = layer.k[0, 1]
            self.kxz[idx] = layer.k[0, 2]
            self.kyz[idx] = layer.k[1, 2]

            self.ux[idx] = layer.ux
            self.uy[idx] = layer.uy

    # get layer index
    def layer_idx(self, z):
        """
        Determine layer index from z-position.

        :param z: z-position
        :returns: Layer index
        """

        if(z < 0.0 or z > self.lg):
            raise ValueError('z outside medium.')

        zs = self.ll[0]
        idx = 0

        while(z > zs):
            idx += 1
            zs += self.ll[idx]

        return idx

    # polar wrapper
    def calc_laplace(self, s, q, phiq, z, z0):
        """
        Compute solution: Polar version
        :param s: Laplace variable
        :param q: Wavevector magnitude
        :param phiq: Wavevector angle
        :param z: Depth
        :param z0: Source layer depth
        :returns: Solution in Fourier-Laplace-space
        """
        # Wrapper: compute wavevector components and call carthesian version
        qx = q*np.cos(phiq)
        qy = q*np.sin(phiq)
        return self.calc_laplace_ft(s, qx, qy, z, z0)

    # Robin boundary conditions
    def calc_laplace_ft(self, s, qx, qy, z, z0):
        """
        Compute solution: Carthesian version
        :param s: Laplace variable
        :param qx: Wavevector x-component
        :param qy: Wavevector y-component
        :param z: Depth
        :param z0: Source layer depth
        :returns: Solution in Fourier-Laplace-space
        """

        # s or (qx and qy) must be scalar.
        assert((not isinstance(s, (collections.Sequence, np.ndarray)))
               or ((not isinstance(qx, (collections.Sequence, np.ndarray)))
                   and (not isinstance(qy, (collections.Sequence, np.ndarray)))))

        # Determine src layer and z layer
        z_layer = self.layer_idx(z)
        z0_layer = self.layer_idx(z0)

        qx2 = qx*qx
        qy2 = qy*qy

        alpha = []
        beta = []
        gamma = []
        en = []

        for i in range(self.Nl):
            alpha.append(
                np.sqrt((self.rho[i]*self.cp[i]*(s+1j*(qx*self.ux[i]+qy*self.uy[i]))
                        + self.kx[i]*qx2 + self.ky[i]*qy2
                        + 2.0*self.kxy[i]*qx*qy) / self.kz[i])
                )
            beta.append((self.kxz[i]*qx + self.kyz[i]*qy)/self.kz[i])
            gamma.append(np.sqrt(alpha[i]*alpha[i] - beta[i]*beta[i]))
            en.append(np.exp(-gamma[i]*self.ll[i]))

        # Build tridiagonal vectors and rhs

        dl = []
        d = []
        du = []
        rhs = []

        # Upper boundary

        z_res = 0.0
        lambda_z = 0.0
        lambda_beta = np.zeros_like(beta[0], dtype=np.complex128)
        lambda_beta_src = np.zeros_like(beta[0], dtype=np.complex128)

        d.append((gamma[0]-self.h1)*en[0])
        du.append(-(gamma[0]+self.h1))

        if(z0_layer == 0):
            rhs.append(-0.5/self.kz[0]
                       * np.exp((1j*beta[0]-gamma[0])*z0)
                       * (1.0-self.h1/gamma[0]))
        else:
            rhs.append(np.zeros_like(gamma[0], dtype=np.complex128))

        # Layer interfaces
        for i in range(self.Nl-1):

            if(i == z_layer):
                z_res = z - lambda_z
                lambda_beta_src = np.copy(lambda_beta)

            lambda_z += self.ll[i]
            lambda_beta += beta[i]*self.ll[i]

            dl.append(self.kz[i]*gamma[i]+self.kz[i+1]*gamma[i+1])
            d.append(en[i]*(self.kz[i+1]*gamma[i+1]-self.kz[i]*gamma[i]))
            du.append(-2.0*self.kz[i+1]*gamma[i+1]*en[i+1])

            dl.append(-2.0*en[i]*self.kz[i]*gamma[i])
            d.append(en[i+1]*(self.kz[i]*gamma[i]-self.kz[i+1]*gamma[i+1]))
            du.append(self.kz[i]*gamma[i]+self.kz[i+1]*gamma[i+1])

            if(i == z0_layer):
                f = 0.5*np.exp((-1j*beta[i]-gamma[i])*(lambda_z-z0))
                p = np.exp(1j*lambda_beta)
                rhs.append(f*p*(1.0 - self.kz[i+1]*gamma[i+1]/(self.kz[i]*gamma[i])))
                rhs.append(2.0*f*p)
            elif((i+1) == z0_layer):
                f = 0.5*np.exp((1j*beta[i+1]-gamma[i+1])*(z0-lambda_z))
                p = np.exp(1j*lambda_beta)
                rhs.append(2.0*f*p)
                rhs.append(f*p*(1.0 - self.kz[i]*gamma[i]/(self.kz[i+1]*gamma[i+1])))
            else:
                rhs.append(np.zeros_like(gamma[0], dtype=np.complex128))
                rhs.append(np.zeros_like(gamma[0], dtype=np.complex128))

        # Lower boundary

        if(z_layer == self.Nl-1):
            z_res = z - lambda_z
            lambda_beta_src = np.copy(lambda_beta)

        lambda_z += self.ll[-1]
        lambda_beta += beta[-1]*self.ll[-1]

        dl.append(gamma[-1]+self.h2)
        d.append(-en[-1]*(gamma[-1]-self.h2))

        if(z0_layer == self.Nl-1):
            f = 0.5/self.kz[-1]*np.exp((-1j*beta[-1]-gamma[-1])*(lambda_z-z0))
            p = np.exp(1j*lambda_beta)
            rhs.append(f*p*(1.0-self.h2/gamma[-1]))
        else:
            rhs.append(np.zeros_like(gamma[0], dtype=np.complex128))

        # reorder
        dl = np.array(dl, order='F')
        d = np.array(d, order='F')
        du = np.array(du, order='F')
        rhs = np.array(rhs, order='F')

        # result matrix
        result = np.empty_like(gamma[0], dtype=np.complex128)

        # solve
        for idx, res in np.ndenumerate(result):
            slc = (Ellipsis,)+idx
            coeff, info = (
                la.lapack.zgtsv(dl[slc], d[slc], du[slc], rhs[slc],
                                overwrite_dl=1, overwrite_d=1,
                                overwrite_du=1, overwrite_b=1)
                )[3::]

            result[idx] = np.exp(-1j*beta[z_layer][idx]*z_res-1j*lambda_beta_src[idx])/s \
                * (coeff[2*z_layer]*np.exp(gamma[z_layer][idx]*(z_res-self.ll[z_layer]))
                    + coeff[2*z_layer+1]*np.exp(-gamma[z_layer][idx]*z_res))

        # analytic particular solution (only required inside the source layer)
        if(z0_layer == z_layer):
            particular = 0.5/(gamma[z0_layer]*self.kz[z0_layer]*s) \
                * np.exp(-gamma[z0_layer]*np.fabs(z-z0)) \
                * np.exp(-1j*beta[z0_layer]*(z-z0))
            result += particular

        return result
