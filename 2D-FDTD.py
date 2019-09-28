#!/bin/env python3

# 2D FDTD without PML

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

c  = 3E8      # speed of light in meters per second^2

def FDTD2D(media,pulse,Nt):
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


    times = np.arange(0,Nt*dt,dt)

    # all arrays are indexed as [ti,yi,xi] (unless there is no t component, in which case they are indexed as [yi,xi])
    Dz  = np.zeros((Nt,Ny,Nx)) # flux density
    Ez  = np.zeros((Nt,Ny,Nx)) # electric field
    Hx  = np.zeros((Nt,Ny,Nx)) # magnetic field
    Hy  = np.zeros((Nt,Ny,Nx)) # magnetic field
    Iz  = np.zeros((Nt,Ny,Nx))

    # read in the coefficient arrays for the media
    if np.isscalar(media.Gaz) and np.isscalar(media.Gbz):
        Gaz = media.Gaz*np.ones((Ny,Nx))
        Gbz = media.Gbz*np.ones((Ny,Nx))
    else:
        Gaz = media.Gaz
        Gbz = media.Gbz

    # get the pulse and save it
    Pz = pulse.getPulse(times)

    # FDTD loop
    for ti, t in enumerate(times):
        if ti == 0: continue
        # I use a numpy trick here to do h[ti][yi][xi-1], by using np.roll
        # axis = 0 corresponds to the y axis and axis = 1 corresponds to the x axis

        # calculate flux density                                                                               # add pulse #
        Dz[ti] = Dz[ti-1]+0.5*(Hy[ti-1] - np.roll(Hy[ti-1],-1,axis=1) - Hx[ti-1] + np.roll(Hx[ti-1],-1,axis=0)) + Pz[ti]

        # calculate electric field
        Ez[ti] = Gaz*(Dz[ti]-Iz[ti-1])

        # calculate ???
        Iz[ti] = Iz[ti-1] + Gbz*Ez[ti]

        # calculate magnetic field
        Hx[ti] = Hx[ti-1] + 0.5*(Ez[ti] - np.roll(Ez[ti],1,axis=0))
        Hy[ti] = Hy[ti-1] + 0.5*(np.roll(Ez[ti],1,axis=1) - Ez[ti])

    # return arrays
    return Ez, Dz, Hx, Hy

class Media:
    """
        The Media class stores the Gaz and Gbz coefficient
        arrays for the media, which are calculated from
        spacially dependent epsilon (permittivity) and
        sigma (conductivity) coefficients. It also stores
        the size of the media and the spacings (dx and dt).

        For free space epsilon = 1 and sigma = 0.

        The default instance of this class is free space.
    """
    def __init__(self, Nx, Ny, dx, epsilon = 1, sigma = 0):

        epsilon_0 = 8.89E12 # permittivity of free space

        self.shape = (Nx, Ny) # shape of the media
        self.dx    = dx       # spacing
        self.dt    = dx/(2*c) # timestep

        # if epsilon and sigma are constant across all space,
        #   then they need to be made into arrays
        if np.isscalar(epsilon) and np.isscalar(sigma):
             epsilon = epsilon*np.ones((Ny,Nx))
             sigma   = sigma*np.ones((Ny,Nx))

        # if only epsilon is constant across space,
        #   then it will be made into an array with the
        #   same shape as sigma
        elif np.isscalar(epsilon):
            assert(sigma.shape==(Ny,Nx))
            epsilon = epsilon*np.ones(sigma.shape)

        # if only sigma is constant across space,
        #   then it will be made into an array with the
        #   same shape as epsilon
        elif np.isscalar(sigma):
            assert(epsilon.shape==(Ny,Nx))
            sigma = sigma*np.ones(epsilon.shape)

        # if both epsilon and sigma are spacially dependent
        #   then Gaz and Gbz can immediately be calculated
        else:
            assert(sigma.shape   == (Ny,Nx))
            assert(epsilon.shape == (Ny,Nx))

        self.Gaz = 1/(epsilon+(sigma*self.dt/epsilon_0))
        self.Gbz = sigma*self.dt/epsilon_0

class Pulse:
    """
        The Pulse class defines a pulse to set as the electric flux in the simulation.
        It creates a 2D gaussian pulse in both time and space
        It is definted by:
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
    def __init__(self, media, radial_center, spacial_spread, temporal_center, temporal_spread):

        self.temporal_center = temporal_center
        self.temporal_spread = temporal_spread

        Ny, Nx = media.shape
        dx     = media.dx

        x, y    = np.meshgrid(np.linspace(-Nx/2,Nx/2,Nx)*dx, np.linspace(-Ny/2,Ny/2,Ny)*dx)
        d       = np.sqrt(x*x+y*y)
        self.g_space = np.exp(-( (d-radial_center)**2 / ( 2.0 * spacial_spread**2 ) ) )

    def getPulse(self, times):
        """
            The getPulse method generates the pulse at the supplied times.
        """

        g_time = np.exp(-(times-self.temporal_center)**2 / (2 * self.temporal_spread**2) )

        return (g_time*(self.g_space*np.ones((g_time.size,self.g_space.shape[0],self.g_space.shape[1]))).T).T

def Animate(F):
    """
        Animate will animate the propagation of a given field
    """

    Nt, Ny, Nx = F.shape

    X = np.arange(Nx)
    Y = np.arange(Ny)

    X,Y = np.meshgrid(X,Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-5,5)

    for ti in range(Nt):
        if ti > 0:
            plot.remove()
        plot = ax.plot_surface(X,Y,F[ti],cmap="viridis")
        plt.draw()
        plt.pause(0.001)

################################################################################

media = Media(200,200,0.1)
pulse = Pulse(media, 30*media.dx, 10*media.dx, 10*media.dt, 2*media.dt)
Ez, Dz, Hx, Hy = FDTD2D(media, pulse, 200)
Animate(Ez)

