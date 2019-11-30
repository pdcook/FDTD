import sys
import numpy as np
import importlib
# automatically add parent-parent directory
sys.path.append([s for s in sys.path if "CodeExamples" in s][0]+"/../../")
# import the module in a very weird way since my filename violates PEP standard
FDTD = importlib.import_module("2D-FDTD-PML")

# diffraction grating geometry

Nx, Ny = 200,200
thickness = 10
epsilon_r = np.ones((Nx, Ny))
sigma = np.zeros((Nx, Ny))

# mask for the diffraction grating, pretty much hard-coded
mask = np.zeros((Nx,Ny),dtype=bool)
mask[Ny//2-thickness//2:Ny//2+thickness//2,:] = True
mask[Ny//2-thickness//2:Ny//2+thickness//2, \
    ::thickness//2] = False
mask[Ny//2-thickness//2:Ny//2+thickness//2, \
    1::thickness//2] = False
mask[Ny//2-thickness//2:Ny//2+thickness//2, \
    ::thickness//2] = False
mask[Ny//2-thickness//2:Ny//2+thickness//2, \
    1::thickness//2] = False

epsilon_r[mask] = 200
sigma[mask] = 500

media = FDTD.Media(Nx, Ny, 0.01, epsilon_r, sigma)
pulse = FDTD.Pulse(media, (0*media.dx,-25*media.dx), \
             0*media.dx, 2*media.dx, 10*media.dt, 2*media.dt)
Ez, Dz, Hx, Hy = FDTD.FDTD2D(media, pulse, 500, PMLSize = 40)
E = FDTD.Energy(Ez,Hx,Hy)
FDTD.Animate(Ez,"Electric Field, $\\tilde{E}_z(x,y)$",cmin=-0.02,cmax=0.02)
