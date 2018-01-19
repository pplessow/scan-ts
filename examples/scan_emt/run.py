#! /usr/bin/env python
from ase.io import *
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.constraints import stretchcombo
from ase.calculators.emt import EMT
import numpy as np

xx=np.linspace(-1,1,100)
yy=[]
for diff in xx:
  
  ## made-up example to work with the EMT 
  atoms=  Atoms('H4', [(0, 0, 0), (3, 0, 0),(2,0,0), (4, 0, 0)])
  atoms.set_constraint([FixAtoms([0,1])])
  
  calc = EMT()
  atoms.set_calculator(calc)
  
  # specifiy constraints through a weighted list of bonds.
  liste=[]
  liste+=[[0,2, 1.0]]  # H1-H2
  liste+=[[1,2,-1.0]]  # H2-H3
  
  # specify intial value of the constraint
  # any string will cause the constraint to keep the current value. otherwise the constraint will be set to the desired value such as 0.1 etc.'
  initial_value=diff
  
  # define constraint
  cc=stretchcombo(initial_value, liste, atoms)
  # adjust structure according to the intial value (in this example the intial value is not modified).
  atoms.set_positions(cc.return_adjusted_positions())
  # set all constrainsts, keeping old constraints 
  atoms.set_constraint([cc]+atoms.constraints)

  dyn = BFGS(atoms)
  dyn.run(fmax=0.0001)
#  atoms.write('it_'+str(diff)+'.xyz')
  yy+=[atoms.get_potential_energy()]

f=open('scan.txt','w')
for x,y in zip(xx,yy):
  f.write(str(x)+' '+str(y)+'\n')
f.close()
print 'max at',xx[np.argsort(yy)[-1]],yy[np.argsort(yy)[-1]]

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.plot(xx,yy,'o-')
plt.title('max at ('+str(xx[np.argsort(yy)[-1]])+', '+str(yy[np.argsort(yy)[-1]])+')')
plt.savefig('scan.pdf')
