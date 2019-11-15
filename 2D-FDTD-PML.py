#!/bin/env python3

# 2D FDTD without PML

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

c  = 3E8      # speed of light in meters per second^2

def FDTD2D(media,pulse,Nt,PMLSize = 5):
    """
        FDTD2D takes a Media object, Pulse object, and the number of timesteps.
        Using the Finite Difference Time Domain method, it propagates the
        electric and magnetic fields forward in time.

        Returns:
            Ez - electric field in the z direction as a function of time and space
            Dz - electric flux in the z direction as a function of time and space
            Hx - magnetic field in the x direction as a function of time and space
            Hy - magnetic field in the y direction as a function of time and space
    """

    Ny, Nx = media.shape
    dx     = media.dx
    dt     = media.dt

    assert(Nx >= 2*PMLSize and Ny >= 2*PMLSize)

    times = np.arange(0,Nt*dt,dt)

    # all arrays are indexed as [ti,yi,xi] (unless there is no t component, in which case they are indexed as [yi,xi])
    Dz  = np.zeros((Nt,Ny,Nx)) # flux density
    Ez  = np.zeros((Nt,Ny,Nx)) # electric field
    Hx  = np.zeros((Nt,Ny,Nx)) # magnetic field
    Hy  = np.zeros((Nt,Ny,Nx)) # magnetic field
    Ihx = np.zeros((Nt,Ny,Nx)) # integral of dyE
    Ihy = np.zeros((Nt,Ny,Nx)) # integral of dxE
    Iz  = np.zeros((Nt,Ny,Nx)) # integral of Ez

    # read in the coefficient arrays for the media
    if np.isscalar(media.Gaz) and np.isscalar(media.Gbz):
        Gaz = media.Gaz*np.ones((Ny,Nx))
        Gbz = media.Gbz*np.ones((Ny,Nx))
    else:
        Gaz = media.Gaz
        Gbz = media.Gbz

    # PML arrays
    n1 = (1/(3*PMLSize**3))*np.arange(1,PMLSize+1)**3
    n2 = (1/(3*PMLSize**3))*(np.arange(1,PMLSize+1)+1/2)**3

    fx1 = np.zeros((Ny,Nx))
    fx1[:,:PMLSize] = n1[::-1]
    fx1[:,-1*PMLSize:] = n1

    fx2 = np.ones((Ny,Nx))
    fx2[:,:PMLSize] = 1/(1+n2[::-1])
    fx2[:,-1*PMLSize:] = 1/(1+n2)

    fx3 = np.ones((Ny,Nx))
    fx3[:,:PMLSize] = (1-n2[::-1])/(1+n2[::-1])
    fx3[:,-1*PMLSize:] = (1-n2)/(1+n2)

    gx2 = 1/(1+fx1)
    gx3 = (1-fx1)/(1+fx1)

    fy1 = np.zeros((Ny,Nx))
    fy1.T[:,:PMLSize] = n1[::-1]
    fy1.T[:,-1*PMLSize:] = n1

    fy2 = np.ones((Ny,Nx))
    fy2.T[:,:PMLSize] = 1/(1+n2[::-1])
    fy2.T[:,-1*PMLSize:] = 1/(1+n2)

    fy3 = np.ones((Ny,Nx))
    fy3.T[:,:PMLSize] = (1-n2[::-1])/(1+n2[::-1])
    fy3.T[:,-1*PMLSize:] = (1-n2)/(1+n2)

    gy2 = 1/(1+fy1)
    gy3 = (1-fy1)/(1+fy1)

    # get the pulse and save it
    Pz = pulse.getPulse(times)

    # FDTD loop
    for ti, t in enumerate(times):
        if ti == 0: continue
        # I use a numpy trick here to do h[ti][yi][xi-1], by using np.roll
        # axis = 0 corresponds to the y axis and axis = 1 corresponds to the x axis

        # calculate flux density                                                                                               # add pulse #
        Dz[ti] = gx3*gy3*Dz[ti-1]+0.5*gx2*gy2*(Hy[ti-1] - np.roll(Hy[ti-1],-1,axis=1) - Hx[ti-1] + np.roll(Hx[ti-1],-1,axis=0)) + Pz[ti]

        # calculate electric field
        Ez[ti] = Gaz*(Dz[ti]-Iz[ti-1])

        # calculate integral of Ez
        Iz[ti] = Iz[ti-1] + Gbz*Ez[ti]

        # calculate magnetic field
        dxE     = np.roll(Ez[ti],1,axis=1) - Ez[ti]
        Ihy[ti] = Ihy[ti-1] + dxE
        Hy[ti]  = fx3*Hy[ti-1] + 0.5*fx2*dxE + fy1*Ihy[ti]

        dyE     = np.roll(Ez[ti],1,axis=0) - Ez[ti]
        Ihx[ti] = Ihx[ti-1] + dyE
        Hx[ti]  = fy3*Hx[ti-1] - 0.5*fy2*dyE - fx1*Ihx[ti]

    # return arrays
    return Ez, Dz, Hx, Hy

class Media:
    """
        The Media class stores the Gaz and Gbz coefficient
        arrays for the media, which are calculated from
        spacially dependent epsilon_r (relative permittivity) and
        sigma (conductivity) coefficients. It also stores
        the size of the media and the spacings (dx and dt).

        For free space epsilon_r = 1 and sigma = 0.

        The default instance of this class is free space.
    """
    def __init__(self, Nx, Ny, dx, epsilon_r = 1, sigma = 0):

        epsilon_0 = 8.89E-12 # permittivity of free space

        self.shape = (Nx, Ny) # shape of the media
        self.dx    = dx       # spacing
        self.dt    = dx/(2*c) # timestep

        # if epsilon_r and sigma are constant across all space,
        #   then they need to be made into arrays
        if np.isscalar(epsilon_r) and np.isscalar(sigma):
             epsilon_r = epsilon_r*np.ones((Ny,Nx))
             sigma   = sigma*np.ones((Ny,Nx))

        # if only epsilon_r is constant across space,
        #   then it will be made into an array with the
        #   same shape as sigma
        elif np.isscalar(epsilon_r):
            assert(sigma.shape==(Ny,Nx))
            epsilon_r = epsilon_r*np.ones(sigma.shape)

        # if only sigma is constant across space,
        #   then it will be made into an array with the
        #   same shape as epsilon_r
        elif np.isscalar(sigma):
            assert(epsilon_r.shape==(Ny,Nx))
            sigma = sigma*np.ones(epsilon_r.shape)

        # if both epsilon_r and sigma are spacially dependent
        #   then Gaz and Gbz can immediately be calculated
        else:
            assert(sigma.shape   == (Ny,Nx))
            assert(epsilon_r.shape == (Ny,Nx))

        self.Gaz = 1/(epsilon_r+(sigma*self.dt/epsilon_0))
        self.Gbz = sigma*self.dt/epsilon_0

class Pulse:
    """
        The Pulse class defines a pulse to set as the electric flux in the simulation.
        It creates a 2D gaussian pulse in both time and space
        It is definted by:
            center          - where the pulse will be centered in space, (x0,y0)
            radial_center   - where the pulse will peak radially from the center
                                for example:
                                radial_center = 0:          radial_center > 0:

                                --------                    --------
                                --------                    --XXXX--
                                ---XX---                    --X--X--
                                ---XX---                    --X--X--
                                --------                    --XXXX--
                                --------                    --------

            spacial_spread  - the standard deviation of the gaussian in space

            temporal_center - where the peak of the pulse will be in time

            temporal_spread - the standard deviation of the gaussian in time

    """
    def __init__(self, media, center, radial_center, spacial_spread, temporal_center, temporal_spread):

        self.temporal_center = temporal_center
        self.temporal_spread = temporal_spread

        Ny, Nx = media.shape
        dx     = media.dx
        x0, y0 = center

        x, y    = np.meshgrid(np.linspace(-Nx/2,Nx/2,Nx)*dx, np.linspace(-Ny/2,Ny/2,Ny)*dx)
        d       = np.sqrt((x-x0)**2+(y-y0)**2)
        self.g_space = np.exp(-( (( d - radial_center )**2) / ( 2.0 * spacial_spread**2 ) ) )

    def getPulse(self, times):
        """
            The getPulse method generates the pulse at the supplied times.
        """

        g_time = np.exp(-(times-self.temporal_center)**2 / (2 * self.temporal_spread**2) )

        return (g_time*(self.g_space*np.ones((g_time.size,self.g_space.shape[0],self.g_space.shape[1]))).T).T

def Animate(F, excludePML = 0, cmin = None, cmax = None):
    """
        Animate will animate the propagation of a given field
    """

    if excludePML: F = F[:,excludePML:-excludePML,excludePML:-excludePML]

    Nt, Ny, Nx = F.shape

    fig, ax = plt.subplots(1,1)

    if cmin is None: cmin = np.min(F)/3
    if cmax is None: cmax = np.max(F)/3
    im = ax.imshow(F[0], animated=True, clim=(cmin,cmax))

    def updatefig(ti):
        ti %= Nt
        im.set_array(F[ti])
        return [im]

    ani = animation.FuncAnimation(fig, updatefig, interval=5, blit=True)
    plt.show()

def Energy(Ez,Hx,Hy):
    """
        Calculate the energy in the domain
    """
    Nt, Ny, Nx = Ez.shape
    E_Ez = np.sum(Ez**2, axis=(1,2))
    E_Hx = np.sum(Hx**2, axis=(1,2))
    E_Hy = np.sum(Hy**2, axis=(1,2))
    return E_Ez+E_Hx+E_Hy

################################################################################

media = Media(200,200,0.1)
pulse = Pulse(media, (0*media.dx,-25*media.dx), 10*media.dx, 10*media.dx, 10*media.dt, 2*media.dt)
Ez, Dz, Hx, Hy = FDTD2D(media, pulse, 200, PMLSize = 10)
E = Energy(Ez,Hx,Hy)
Animate(Ez)
