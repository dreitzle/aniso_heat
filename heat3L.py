# coding: utf-8

"""heat3L.py: Explicit 3-layered solution implementation"""

import numpy as np

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2018, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"


# Isolating boundaries determinants

def _calc_M_3L_delta_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                               p1, p2, p3, m1, m2, m3):
    g1 = a2*k2*p2*p3 + a3*k3*m2*m3
    g2 = a2*k2*m2*p3 + a3*k3*p2*m3
    return a1*a3*(a1*k1*m1*g1 + a2*k2*p1*g2)


def _calc_A_3L_delta_q1_z1_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                                     p1, p2, p3, m1, m2, m3, f1, f2):
    t1 = a2*k2*p3*(a1*k1*p2-a2*k2*m2)
    t2 = a3*k3*m3*(a2*k2*p2-a1*k1*m2)
    return a3/k1*(e1*f1+f2)*(t1-t2)


def _calc_B_3L_delta_q1_z1_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                                     p1, p2, p3, m1, m2, m3, f1, f2):
    t1 = a2*k2*p2*p3+a3*k3*m2*m3
    t2 = a2*k2*m2*p3+a3*k3*p2*m3
    return a1*a3*(f1+e1*f2)*t1+a2*a3*k2/k1*(f1-e1*f2)*t2


def _calc_A_3L_delta_q1_z2_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                                     p1, p2, p3, m1, m2, m3, f1, f2):
    return 2.0*a1*a3*e2*(e1*f1+f2)*(a2*k2*p3-a3*k3*m3)


def _calc_B_3L_delta_q1_z2_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                                     p1, p2, p3, m1, m2, m3, f1, f2):
    return 2.0*a1*a3*(e1*f1+f2)*(a2*k2*p3+a3*k3*m3)


def _calc_A_3L_delta_q1_z3_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                                     p1, p2, p3, m1, m2, m3, f1, f2):
    return 4.0*a1*a2*a3*e2*e3*k2*(e1*f1+f2)


def _calc_B_3L_delta_q1_z3_isolating(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                                     p1, p2, p3, m1, m2, m3, f1, f2):
    return 4.0*a1*a2*a3*e2*k2*(e1*f1+f2)


# Robin boundaries determinants

def _calc_M_3L_delta(k1, k2, k3, a1, a2, a3, e1, e2, e3, p1, p2, p3,
                     m1, m2, m3, h1, h2):
    g1 = a2*k2*p2*(a3*p3+h2*m3)
    g2 = a3*k3*m2*(a3*m3+h2*p3)
    g3 = a2*k2*m2*(a3*p3+h2*m3)
    g4 = a3*k3*p2*(a3*m3+h2*p3)
    g5 = a2*k2*(a3*p3+h2*m3)*(h1*k1*p2+a2*k2*m2)
    g6 = a3*k3*(a3*m3+h2*p3)*(h1*k1*m2+a2*k2*p2)
    return a1*a1*k1*m1*(g1+g2)+a2*k2*h1*m1*(g3+g4)+a1*p1*(g5+g6)


def _calc_A_3L_delta_q1_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a2*k2*(a3*p3+h2*m3)*(a1*k1*p2-a2*k2*m2)
    t2 = a3*k3*(a3*m3+h2*p3)*(a1*k1*m2-a2*k2*p2)
    return (e1*f1+(1.0+h1/a1)*f2)*(t1+t2)/k1


def _calc_B_3L_delta_q1_z1(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a1*(f1+e1*f2)-h1*e1*f2
    t2 = a1*(f1-e1*f2)+h1*e1*f2
    t3 = a1*k1*p2*t1+a2*k2*m2*t2
    t4 = a1*k1*m2*t1+a2*k2*p2*t2
    return (a2*k2*(a3*p3+h2*m3)*t3 + a3*k3*(a3*m3+h2*p3)*t4)/(a1*k1)


def _calc_A_3L_delta_q1_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a2*k2*(a3*p3+h2*m3)
    t2 = a3*k3*(a3*m3+h2*p3)
    return 2.0*e2*(a1*(e1*f1+f2)+h1*f2)*(t1-t2)


def _calc_B_3L_delta_q1_z2(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    t1 = a2*k2*(a3*p3+h2*m3)
    t2 = a3*k3*(a3*m3+h2*p3)
    return 2.0*(a1*(e1*f1+f2)+h1*f2)*(t1+t2)


def _calc_A_3L_delta_q1_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return 4.0*a2*k2*e2*e3*(a3-h2)*(a1*(e1*f1+f2)+h1*f2)


def _calc_B_3L_delta_q1_z3(k1, k2, k3, a1, a2, a3, e1, e2, e3,
                           p1, p2, p3, m1, m2, m3, h1, h2, f1, f2):
    return 4.0*a2*k2*e2*(a3+h2)*(a1*(e1*f1+f2)+h1*f2)


# default source functions

def _src_switch(s):
    return 1.0/s


def _Text_null(s, qx, qy):
    return 0.0


# solution implementation

class heat_3L_3D_aniso:
    """Explicit 3-layered solution implementation."""

    def __init__(self, rho1, cp1, k1m, rho2, cp2, k2m, rho3, cp3, k3m,
                 l1, l2, l3, h1=0, h2=0, ux=0, uy=0,
                 f_src=_src_switch, f_Text=_Text_null):
        """
        :param rho1: Mass density of the first layer.
        :param cp1: Heat capacity of the first layer.
        :param k1m: Thermal conductivity of the first layer
            (3x3 symmetric Matrix).
        :param rho2: Mass density of the second layer.
        :param cp2: Heat capacity of the second layer.
        :param k2m: Thermal conductivity of the second layer
            (3x3 symmetric Matrix).
        :param rho3: Mass density of the third layer.
        :param cp3: Heat capacity of the third layer.
        :param k3m: Thermal conductivity of the third layer
            (3x3 symmetric Matrix).
        :param l1: Thickness of the first layer.
        :param l2: Thickness of the second layer.
        :param l3: Thickness of the third layer.
        :param h1: Heat transfer coefficient of upper boundary.
        :param h2: Heat transfer coefficient of lower boundary.
        :param ux: x-component of the medium velocity (all layers).
        :param uy: y-component of the medium velocity (all layers).
        :param f_src: Time dependence of the source (Laplace transform).
        :param f_Text: Time and space dependence of the upper fluid
            temperature (Laplace and Fourier transform).
        """
        assert isinstance(k1m, np.ndarray)
        assert isinstance(k2m, np.ndarray)
        assert isinstance(k3m, np.ndarray)

        assert k1m.shape == (3, 3)
        assert k2m.shape == (3, 3)
        assert k3m.shape == (3, 3)

        assert np.allclose(k1m, k1m.T, atol=1e-8)
        assert np.allclose(k2m, k2m.T, atol=1e-8)
        assert np.allclose(k3m, k3m.T, atol=1e-8)

        self.rho1 = rho1
        self.cp1 = cp1
        self.k1x = k1m[0, 0]
        self.k1y = k1m[1, 1]
        self.k1z = k1m[2, 2]
        self.k1xy = k1m[0, 1]
        self.k1xz = k1m[0, 2]
        self.k1yz = k1m[1, 2]
        self.u1x = ux
        self.u1y = uy

        self.rho2 = rho2
        self.cp2 = cp2
        self.k2x = k2m[0, 0]
        self.k2y = k2m[1, 1]
        self.k2z = k2m[2, 2]
        self.k2xy = k2m[0, 1]
        self.k2xz = k2m[0, 2]
        self.k2yz = k2m[1, 2]
        self.u2x = ux
        self.u2y = uy

        self.rho3 = rho3
        self.cp3 = cp3
        self.k3x = k3m[0, 0]
        self.k3y = k3m[1, 1]
        self.k3z = k3m[2, 2]
        self.k3xy = k3m[0, 1]
        self.k3xz = k3m[0, 2]
        self.k3yz = k3m[1, 2]
        self.u3x = ux
        self.u3y = uy

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.h1 = h1
        self.h2 = h2

        self.lg = l1+l2+l3

        if f_src is not None:
            self.f_src = f_src
        else:
            self.f_src = lambda s: 0.0

        self.f_Text = f_Text

    # Polar wrapper
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
        # Wrapper: compute wavevector components and
        # call carthesian version
        qx = q*np.cos(phiq)
        qy = q*np.sin(phiq)
        return self.calc_laplace_ft(s, qx, qy, z, z0)

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
        if(z < 0.0 or z > self.lg):
            raise ValueError('z outside medium.')
        if(z0 < 0.0 or z0 > self.lg):
            raise ValueError('z0 outside medium.')

        qx2 = qx*qx
        qy2 = qy*qy

        alpha1 = np.sqrt((self.rho1*self.cp1*(s+1j*(qx*self.u1x+qy*self.u1y))
                          + self.k1x*qx2 + self.k1y*qy2
                          + 2.0*self.k1xy*qx*qy) / self.k1z)
        alpha2 = np.sqrt((self.rho2*self.cp2*(s+1j*(qx*self.u2x+qy*self.u2y))
                          + self.k2x*qx2 + self.k2y*qy2
                          + 2.0*self.k2xy*qx*qy) / self.k2z)
        alpha3 = np.sqrt((self.rho3*self.cp3*(s+1j*(qx*self.u3x+qy*self.u3y))
                          + self.k3x*qx2 + self.k3y*qy2
                          + 2.0*self.k3xy*qx*qy) / self.k3z)

        beta1 = (self.k1xz*qx + self.k1yz*qy)/self.k1z
        beta2 = (self.k2xz*qx + self.k2yz*qy)/self.k2z
        beta3 = (self.k3xz*qx + self.k3yz*qy)/self.k3z

        gamma1 = np.sqrt(alpha1*alpha1 - beta1*beta1)
        gamma2 = np.sqrt(alpha2*alpha2 - beta2*beta2)
        gamma3 = np.sqrt(alpha3*alpha3 - beta3*beta3)

        e1 = np.exp(-gamma1*self.l1)
        e2 = np.exp(-gamma2*self.l2)
        e3 = np.exp(-gamma3*self.l3)

        p1 = (1.0 + e1*e1)
        p2 = (1.0 + e2*e2)
        p3 = (1.0 + e3*e3)

        m1 = (1.0 - e1*e1)
        m2 = (1.0 - e2*e2)
        m3 = (1.0 - e3*e3)

        detM = _calc_M_3L_delta(self.k1z, self.k2z, self.k3z,
                                gamma1, gamma2, gamma3, e1, e2, e3,
                                p1, p2, p3, m1, m2, m3, self.h1, self.h2)

        if(z0 < self.l1):

            qb = np.exp(1j*beta1*z0)

            f1 = 0.5*self.f_src(s)*qb*np.exp(-gamma1*z0)*(1.0-self.h1/gamma1) \
                + self.h1*self.k1z*self.f_Text(s, qx, qy)
            f2 = 0.5*self.f_src(s)*qb*np.exp(-gamma1*(self.l1-z0))

            if(z < self.l1):
                detA = _calc_A_3L_delta_q1_z1(
                    self.k1z, self.k2z, self.k3z, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1, f2)
                detB = _calc_B_3L_delta_q1_z1(
                    self.k1z, self.k2z, self.k3z, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1, f2)
                particular = 0.5*self.f_src(s)/(gamma1*self.k1z) \
                    * np.exp(-gamma1*np.fabs(z-z0))
                return ((detA*np.exp(gamma1*(z-self.l1))
                        + detB*np.exp(-gamma1*z))/detM + particular) \
                    * np.exp(-1j*beta1*z)
            elif(z < self.l1+self.l2):
                detA = np.exp(-1j*beta1*self.l1)*_calc_A_3L_delta_q1_z2(
                    self.k1z, self.k2z, self.k3z, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1, f2)
                detB = np.exp(-1j*beta1*self.l1)*_calc_B_3L_delta_q1_z2(
                    self.k1z, self.k2z, self.k3z, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1, f2)
                return (detA*np.exp(gamma2*(z-self.l1-self.l2))
                        + detB*np.exp(-gamma2*(z-self.l1))) \
                    / detM*np.exp(-1j*beta2*(z-self.l1))
            else:
                detA = np.exp(-1j*(beta1*self.l1+beta2*self.l2)) \
                    * _calc_A_3L_delta_q1_z3(
                        self.k1z, self.k2z, self.k3z, gamma1, gamma2, gamma3,
                        e1, e2, e3, p1, p2, p3, m1, m2, m3,
                        self.h1, self.h2, f1, f2)
                detB = np.exp(-1j*(beta1*self.l1+beta2*self.l2)) \
                    * _calc_B_3L_delta_q1_z3(
                        self.k1z, self.k2z, self.k3z, gamma1, gamma2, gamma3,
                        e1, e2, e3, p1, p2, p3, m1, m2, m3,
                        self.h1, self.h2, f1, f2)
                return (detA*np.exp(gamma3*(z-self.lg))
                        + detB*np.exp(-gamma3*(z-self.l1-self.l2))) \
                    / detM*np.exp(-1j*beta3*(z-self.l1-self.l2))

        elif(z0 < self.l1+self.l2):
            raise NotImplementedError
        else:
            raise NotImplementedError
