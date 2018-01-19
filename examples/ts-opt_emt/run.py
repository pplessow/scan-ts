#! /usr/bin/env python
from ase.io import write
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import stretchcombo
from ase.constraints import FixAtoms
from ase.optimize.arpess import arpess


calc = EMT()


## made-up example to work with the EMT 
atoms=  Atoms('H4', [(0, 0, 0), (3, 0, 0),(2,0,0), (4, 0, 0)])
atoms.set_constraint([FixAtoms([0,1])])
atoms.write('initial.xyz')

calc = EMT()
atoms.set_calculator(calc)

# specifiy constraints through a weighted list of bonds.
liste=[]
liste+=[[0,2, 1.0]]  # H1-H2
liste+=[[1,2,-1.0]]  # H2-H3

# specify intial value of the constraint
# any string will cause the constraint to keep the current value. otherwise the constraint will be set to the desired value such as 0.1 etc.'
initial_value=1.0

# define constraint
cc=stretchcombo(initial_value, liste, atoms)
# adjust structure according to the intial value (in this example the intial value is not modified).
atoms.set_positions(cc.return_adjusted_positions())
# set all constrainsts, keeping old constraints 
atoms.set_constraint([cc]+atoms.constraints)

dyn = arpess(atoms, scanned_constraint=cc) # minimum: scanned constraint needs to be specified.
dyn.run(fmax=0.001)
atoms.write('final.xyz')
