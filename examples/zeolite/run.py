#! /usr/bin/env python
from ase.constraints import stretchcombo
from ase.optimize.arpess import arpess
from ase.io import read
import ase.calculators.vasp as vasp_calculator

atoms=read('POSCAR')

# specifiy constraints through a weighted list of bonds.
liste=[]
liste+=[[116,108,1.0]] # O1-C
liste+=[[108,94,-1.0]] # C-O2 

# specify intial value of the constraint
# any string will cause the constraint to keep the current value. otherwise the constraint will be set to the desired value such as 0.1 etc.'
initial_value=' '

# define constraint
cc=stretchcombo(initial_value, liste, atoms)
# adjust structure according to the intial value (in this example the intial value is not modified).
atoms.set_positions(cc.return_adjusted_positions()) 
# set all constrainsts, keeping old constraints (there are no old constraints in this example).  
atoms.set_constraint([cc]+atoms.constraints)

calc = vasp_calculator.Vasp(encut=400,
                        xc='PBE',
                        ivdw=11,
                        gga='PE',
                        kpts  = (1,1,1),
                        ncore=4,
                        gamma = True,
                        ismear=0,
                        nelm=250,
                        algo = 'fast',
                        sigma = 0.1,
                        ibrion=-1,
                        ediffg=-0.01,
                        ediff=1e-8,
                        prec='normal',
                        nsw=50,
                        lreal='Auto',
                        ispin=1)

atoms.set_calculator(calc)
dyn = arpess(atoms, scanned_constraint=cc) # minimum: scanned constraint needs to be specified.
dyn.run(fmax=0.01)
atoms.write('final.cif')
