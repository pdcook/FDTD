import sys
import importlib
# automatically add parent-parent directory
sys.path.append([s for s in sys.path if "CodeExamples" in s][0]+"/../../")
# import the module in a very weird way since my filename violates PEP standard
FDTD = importlib.import_module("2D-FDTD-PML")

# free-space simulation with large PML

media = FDTD.Media(200,200,0.1)
pulse = FDTD.Pulse(media, (0*media.dx,-25*media.dx), \
             10*media.dx, 10*media.dx, 10*media.dt, 2*media.dt)
Ez, Dz, Hx, Hy = FDTD.FDTD2D(media, pulse, 500, PMLSize = 30)
E = FDTD.Energy(Ez,Hx,Hy)
FDTD.Animate(Ez,"Electric Field, $\\tilde{E}_z(x,y)$")
