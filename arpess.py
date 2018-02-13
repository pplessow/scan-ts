# -*- coding: utf-8 -*-
import pickle
import os
import sys
import numpy as np

from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS
from ase.io import read,write

class scan_bfgs(BFGS):
    def pass_constraint(self, scanned_constraint=None,totfmax=0.05):
        self.scanned_constraint = scanned_constraint
        self.totfmax=totfmax
    def converged(self, forces=None):
        """Did the optimization converge?"""
        fmax_full = max([np.linalg.norm(f_1+f_2) for f_1,f_2 in zip (forces,self.scanned_constraint.projected_forces)])
        pg = self.scanned_constraint.projected_force
        f_thresh = min(0.1,abs(self.fmax * self.scanned_constraint.f_thresh))

        if forces is None:
            forces = self.atoms.get_forces()
        fmax_current = np.sqrt((forces**2).sum(axis=1).max())

        print '>>>>>> convergence-criterion is',f_thresh,'based on proj. force',pg,'f_cur=',fmax_current
        if self.nsteps < 2:
          print '>>> require at least 3 geometry steps, current step is',self.nsteps
          return False

        if  fmax_full < self.totfmax:
          print 'total max atomic gradient is below threshold:',fmax_full,'<',self.totfmax
          return True

        if fmax_current < f_thresh:
          print 'max atomic gradient is below threshold:',fmax_current,'<',f_thresh
          return True


class arpess(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=0.2, master=None,scanned_constraint=None,constraintlist=None,a=None,
                 auto_threshold=True,fixed_conv_ratio=0.8,
                 max_interval_steps=3,interval_step=0.5,
                 log_prefix='',traj_prefix='',
                 adaptive_threshold=0.6,linear_interpol=False,do_cubic=None):
        """ TS Optimization with constraints
            Class for performing 1D-scan maximization along one constraint, while minimizing all
            remaining degrees of freedom.
            The scanned constraint (XX) can be passed in two ways:
            1) (Recommended) scanned_constraint=XX
            2) Has to be the FIRST constraint in the full list of constraints:
               constraintlist = [ XX, other constraints]
            Addditional constraints, such as FixAtoms, will be kept fixed
            Tested only for 
              - scanned_constraint = stretchcombo
              - additional constraints = FixAtoms
            Should work with other additional constraints. 
            To scan other constraints, probably these constraints will require additional
            functions that were specifically added to stretchcombo.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        print '********************************************************************************'
        print '*                                    ARPESS                                    *' 
        print '*               Automated Relaxed Potential Energy Surface Scans               *' 
        print '*  Plessow, P.N. J. Chem. Theory Comput. 2018, 14, 981−990.                    *'
        print '********************************************************************************'
        if maxstep is not None:
            if maxstep > 1.0:
                raise ValueError('You are using a much too large value for ' +
                                 'the maximum step size: %.1f Å' % maxstep)
            self.maxstep = maxstep
        if constraintlist != None:
          self.constraintlist = constraintlist
          self.only_other_constraints = [self.constraintlist[-ii] for ii in range (1,len(self.constraintlist))]
          self.scanned_constraint = self.constraintlist[0]
        else:
          self.scanned_constraint = scanned_constraint
          self.constraintlist = atoms.constraints
          self.only_other_constraints = [const for const in self.constraintlist if const != self.scanned_constraint]
        print 'scan along constraint',self.scanned_constraint
        print 'keep constraints',self.only_other_constraints
        self.a = self.scanned_constraint.a
        self.na_step = 0
        self.atoms=atoms
        self.maxstep = maxstep
        self.maxpos_last = 1e-9
        self.max_interval_steps = max_interval_steps
        self.interval_step = interval_step
        self.adaptive_threshold = adaptive_threshold
        self.linear_interpol = linear_interpol
        self.auto_threshold=auto_threshold
        self.fixed_conv_ratio=fixed_conv_ratio
        self.log_prefix=log_prefix
        self.traj_prefix=traj_prefix
        if linear_interpol not in [None,False,True]:
          try:
            self.max_linear_interpol = float(linear_interpol)
            self.linear_interpol = True
          except:
            print 'not sure what to do with linear_interpol=',linear_interpol
            sys.exit(1)
        else:
          self.max_linear_interpol = maxstep
        self.do_cubic = do_cubic
        if do_cubic not in [None,False,True]:
          try:
            self.do_cubic  = max (do_cubic, 1.0/do_cubic)
          except:
            print 'not sure what to do with do_cubic=',do_cubic
            sys.exit(1)
        print 'do_cubic parameter is',do_cubic

    def run(self, fmax=0.01, steps=100000000):
        """Run structure optimization algorithm.
        *steps*."""


        self.fmax = fmax
        step = 0
        self.interval_steps = 0
        self.dopt={}
        self.read_restart()
        flog=open(self.log_prefix+'scan.log','w')
        flog.write('Force is evaluated as max(f_i) where i the cartesian force on atom i   \n')
        flog.write('--> the maximum components of proj., full and (full-proj.) force are   \n')
        flog.write('    not necessarily the same and these components therefore dont add up\n')
        flog.write('units are eV and eV/Angstrom                                           \n')
        flog.write('-----------------------------------------------------------------------------------\n')
        flog.write('Step         a         Energy       proj. force  full force   (full-proj.) force\n')
        flog.write('-----------------------------------------------------------------------------------\n')
        while step < steps:
            print 'step',step,'performing constrained optimization at a =',self.a
            e,pg,fi_max,f = self.do_constrained_optimization()
            f_max = max([np.linalg.norm(ff) for ff in f])

            flog.write('   '+' '.join([str(step)]+['%12.4f' % ii for ii in [self.a,e,pg,fi_max,f_max]])+'\n')
            flog.flush()
            self.converged()
            self.step()
            step += 1
            self.na_step += 1
 
        flog.close()

    def restart_tighten_scan(self):
        if self.adaptive_threshold != None:
          print 'switching from adaptive threshold to constant threshold'
          print '  convergence criterion for constrained optimization is = ',self.fmax * self.fixed_conv_ratio
          self.adaptive_threshold = None
        elif self.fixed_conv_ratio > 0.1:
          self.fixed_conv_ratio=0.1
          print 'lowering convergence criterion for constrained optimization to = ',self.fmax * self.fixed_conv_ratio
        else:
          print 'calculation not behaved. giving up.'
          sys.exit(1)
        os.system('mv scan_restart.pkl scan_restart_before-restart.pkl')
        os.system('mv scan.log scan_before-restart.log')
        return self.run(fmax = self.fmax)

    def converged(self):
        for best_f in self.dopt[self.a]:
          if best_f  < self.fmax:
            print '>>> converged : a = ',self.a,'; total force = ',best_f
            print '>>> write converged structure to ','final.traj'
            self.atoms.set_positions(self.dopt[self.a][best_f]['xyz'])
            self.atoms.write(self.traj_prefix+'final.traj')
            sys.exit(1)
        return False
  
    def write_restart(self):
        with open('scan_restart.pkl', 'wb') as pickel:
          pickle.dump(self.dopt,pickel)

    def read_restart(self):
        if os.path.exists('scan_restart.pkl'):
          print 'try to restart from restart file'
          with open('scan_restart.pkl', 'rb') as pickel:
            self.dopt = pickle.load(pickel)
          for aa in self.dopt:
            for fi_max in self.dopt[aa]:
              self.na_step = max(self.na_step,self.dopt[aa][fi_max]['na_step'])
          self.na_step += 1
          self.step()
     
    def getx_from_na_step(self,na_step):
       for aa in self.dopt:
         for fi_max in self.dopt[aa]:
           if na_step == self.dopt[aa][fi_max]['na_step']:
             return aa,fi_max
       return None,None

    def analyze_data(self):
        xx=[]
        yy=[]
        ff=[]
        ft=[]
        xyzs = []
        froot = None
        maxpos = None
        max2pos = None
        max3pos = None
        is_behaved = True
        for a in sorted([aa for aa in self.dopt]):
          best_f = min([f for f in self.dopt[a]])

          xx   += [a]
          yy   += [ self.dopt[a][best_f]['e'] ]
          ff   += [ self.dopt[a][best_f]['pf'] ]
          xyzs += [ self.dopt[a][best_f]['xyz'] ]
          ft   += [ best_f ]
       


          if len(xx) > 1:
            if ff[-1] * ff[-2] <0:
              if froot == None:
                froot =[ len(ff)-2, len(ff)-1 ]
              else:
                is_behaved = False
        if froot == None: 
####  in case of all-positive/all-negative gradient use outmost points in search direction
          if ff[0] > 0: # positive gradient: Go to smaller a
            maxpos = 0
            if len(xx)>1:  
              max2pos = maxpos+1
            if len(xx)>2:  
              max3pos = maxpos+2
          else:
            maxpos = len(ff)-1
            if len(xx)>1:  
              max2pos = maxpos-1
            if len(xx)>2:  
              max3pos = maxpos-2
####  have root = change in gradient. search here.
        else: 

 ## choose last point as ref. if possible
 ## make sure maxpos is not the same as in last step
          if self.maxpos_last == xx[froot[1]] or self.getx_from_na_step(self.na_step)[0] == xx[froot[0]]:
            maxpos =froot[0]
            max2pos=froot[1]
          elif self.maxpos_last == xx[froot[0]] or self.getx_from_na_step(self.na_step)[0] == xx[froot[1]]:
            maxpos =froot[1]
            max2pos=froot[0]
          elif abs(ff[froot[0]]) < abs(ff[froot[1]]):# maxpos is point with lowest absolute gradient
            maxpos =froot[0]
            max2pos=froot[1]
          else:
            maxpos =froot[1]
            max2pos=froot[0]

          topf=None
          bopf=None
          topfl=[ froot[0]  , froot[1], froot[1]+1 ]
          bopfl=[ froot[0]  , froot[1], froot[0]-1 ]
          if froot[1]+1 < len(ff) and abs(ff[froot[1]+1]) > abs(ff[froot[1]]): 
            topf = abs(ff[froot[1]+1])
          if froot[0]-1 >=0       and abs(ff[froot[0]-1]) > abs(ff[froot[0]]): 
            bopf = abs(ff[froot[0]-1])
          if topf == None:
            if bopf != None:
              froot = bopfl
          else:
            if bopf == None:
              froot = topfl
            elif maxpos < max2pos:
              froot = bopfl
            else:
              froot = topfl

          if len(froot)>2:
            max3pos=froot[2]
          
        return xx, yy, ff, ft, is_behaved, maxpos, max2pos, max3pos,xyzs,froot

    def step(self):

        xx, yy, ff, ft, is_behaved, maxpos, max2pos, max3pos,xyzs,froot = self.analyze_data()
        if not is_behaved:
          print '-------------------------------------------------------------'
          print 'scan has two roots. restart with tighter convergence criteria'
          print '-------------------------------------------------------------'
          self.restart_tighten_scan()
        self.a = xx[maxpos]
        pg = ff[maxpos]
        if max3pos == None:
          if max2pos == None:
            print 'maxpos',xx[maxpos]
          else:
            print 'maxpos, max2pos',xx[maxpos],xx[max2pos]
        else:
          print 'maxpos, max2pos, max3pos =',xx[maxpos],xx[max2pos],xx[max3pos]
        print '       a        E             f               '
        print '--------------------------------------------'
        for x_x, y_y, f_f in zip(xx, yy,ff):
          print '%10.4f' % x_x, '%12.4f' % y_y,'%10.4f' % f_f
        self.maxpos_last = xx[maxpos]

        if self.converged():
            print 'calculation is converged. no optimization step is performed'
            return

        cubic_point = None
        newton_statpoint = None

        step_interval = None
        if ff[maxpos] < 0.0 and len(xx)>maxpos+1:
          step_interval = xx[maxpos+1]
        elif ff[maxpos] > 0.0 and maxpos>0:
          step_interval = xx[maxpos-1]

        primitive_point = xx[maxpos] + -ff[maxpos]

        if len(xx) > 1:
          avg_hess = -np.polyfit(xx,ff,1)[0]
          if max3pos == None or abs(xx[maxpos]-xx[max2pos]) < abs(xx[maxpos]-xx[max3pos]):
            graddiff_hess = -(ff[maxpos]-ff[max2pos])/(xx[maxpos]-xx[max2pos])
          else:
            graddiff_hess = -(ff[maxpos]-ff[max3pos])/(xx[maxpos]-xx[max3pos])
          if graddiff_hess < 0.0:
            if avg_hess < 0.0:
              min_hess = 10.0 * avg_hess
              max_hess = avg_hess / 10.0 
              if graddiff_hess < min_hess:
                print 'limit hessian to 10*average hessian from all points since local hessian from two points is too different.',graddiff_hess,avg_hess
                graddiff_hess = min_hess
              if graddiff_hess > max_hess:
                print 'limit hessian to 0.1*average hessian from all points since local hessian from two points is too different.',graddiff_hess,avg_hess
                graddiff_hess = max_hess
          elif avg_hess < 0.0:
            print 'use average hessian from all points since local hessian has wrong sign',graddiff_hess,avg_hess
            graddiff_hess = avg_hess

          if graddiff_hess < 0.0:
            newton_statpoint = pg / graddiff_hess + self.a
            print 'hess = ',graddiff_hess,'grad = ',pg
            print 'newton step predicts TS at ',newton_statpoint

        if len(xx) > 2 and max3pos !=None and self.do_cubic !=None:
          ## get third-next point --> change to use points enclosing the root, if possible
          sortx=[xx[xxx] for xxx in sorted([maxpos,max2pos,max3pos])]
          sorty=[ff[xxx] for xxx in sorted([maxpos,max2pos,max3pos])]
          p2 = np.polyfit(sortx,sorty,2)
          if newton_statpoint!=None:
            xref = newton_statpoint
          else:
            xref = xx[maxpos]
          cubic_point = self.closest_quadmin(p2,xref)
          try:
            reasonable_ratio = abs((newton_statpoint-self.a)/(cubic_point-self.a))
            if reasonable_ratio > self.do_cubic or reasonable_ratio < 1.0/self.do_cubic:
              print 'do not use cubic step. difference to hessian step too large',cubic_point,newton_statpoint
              cubic_point = None
            else:
              print 'Use cubic step. difference to hessian step reasonable',cubic_point,newton_statpoint
          except:
            print 'failed to do cubic step'
            pass

        print 'cubic',cubic_point
  
        steplist  = [ cubic_point,  newton_statpoint,  primitive_point]
        steplabel = ['cubic_point','newton_statpoint','primitive_point']

        for new_point,steplab in zip(steplist,steplabel):
          if new_point != None:
            a_step = new_point - self.a
            if a_step * pg < 0:
              print 'try step',steplab,'Delta a = ',a_step
              break
            else:
              print 'dismiss step',steplab,'Delta a = ',a_step,'wrong direction'

        old_xyz = xyzs[maxpos]
### this is the step relative to last point which is not necessarily the same as maxpos
        if step_interval != None:
          print 'interval',step_interval,step_interval-self.a,a_step
          if abs(a_step) > abs(step_interval-self.a):

            if self.auto_threshold and self.interval_steps >= self.max_interval_steps:
              print '-------------------------------------------------------------'
              print 'calculated step violates interval from previous steps.'
              print '         restart with tighter convergence criteria'
              print '-------------------------------------------------------------'
              self.restart_tighten_scan()
            else:
              print '-------------------------------------------------------------'
              print 'calculated step violates interval from previous steps.'
              print '         scale back step by ',self.interval_step
              print '-------------------------------------------------------------'
              self.interval_steps += 1
              a_step = self.interval_step * (step_interval - xx[maxpos])

        self.a += a_step
### step-size control according to the cartesian coordinates is applied below, where the cartesians are generated ###
        scaled_back = self.setup_starting_structure(old_xyz)
        if scaled_back != None:
          print 'step scaled corresponds to delta a =',a_step*scaled_back
          self.a -= (1.0-scaled_back) * a_step
        print 'set positions. again.'
        self.atoms.set_positions(self.scanned_constraint.reset_a(self.a,self.atoms))
        print '... done'
      
        return

    def closest_quadmin(self,p2,xref):
      p = p2[1] / p2[0]
      q = p2[2] / p2[0]
      if 0.25*p**2 -q < 0:
        print 'quadratic equation as no root',p2,p,q
        return None
      x1 = -0.5*p + np.sqrt(0.25*p**2 -q)
      x2 = -0.5*p - np.sqrt(0.25*p**2 -q)
      if abs(x1-xref) < abs(x2-xref):
        return x1
      else:
        return x2

    def linear_interpol_structures(self,xyz_temp,a_temp):
        new_xyz = []
        prop_a = (self.a - a_temp[0]) / (a_temp[1] - a_temp[0])
        for xi,xj in zip(xyz_temp[0],xyz_temp[1]):
          new_xyz += [[ xxi + prop_a * (xxj-xxi) for xxi,xxj in zip(xi,xj)]] 


        maxx1 = np.max([np.linalg.norm(x1-x2) for x1,x2 in zip(new_xyz,xyz_temp[0])])
        maxx2 = np.max([np.linalg.norm(x1-x2) for x1,x2 in zip(new_xyz,xyz_temp[1])])
        if min(maxx1,maxx2) > self.max_linear_interpol:
          print 'interpolated max step exceeds interpol limit',min(maxx1,maxx2),'>',self.max_linear_interpol
          return xyz_temp[np.argsort([maxx1,maxx2])[0]],False
        else:
          return np.asarray(new_xyz),True

    def xyz_stepsize_control(self,new_xyz,old_xyz):
        maxx = np.max([np.linalg.norm(x1-x2) for x1,x2 in zip(new_xyz,old_xyz)])
        if maxx > self.maxstep:
          scaled_back = self.maxstep / maxx
          print 'max atomic component of step =',maxx,'exceeds threshold',self.maxstep
          s_xyz = [x2+scaled_back*(x1-x2) for x1,x2 in zip(new_xyz,old_xyz)]
          self.atoms.set_constraint(self.only_other_constraints)
          self.atoms.set_positions(s_xyz)
          print 'reset constraint'
          self.atoms.set_constraint(self.constraintlist)
          return scaled_back
        else:
          print 'max atomic component of step =',maxx,'ok'
          return None

    def setup_starting_structure(self,last_xyz):
        old_xyz = last_xyz
        a_s = [ii for ii in self.dopt]
        xyz_temp=[]
        a_temp = []
        for ordered in np.argsort([abs(ii-self.a) for ii in a_s]):
          for best_f in sorted([best_ff for best_ff in self.dopt[a_s[ordered]]]):
            xyz_temp += [ self.dopt[a_s[ordered]][best_f]['xyz'] ]
            a_temp   += [           a_s[ordered]                 ]
            break
          if self.linear_interpol and len(a_temp) == 2:
            print 'starting structure for a = ',self.a,'is generated by interpolation of cartesian coordiates of a1,a2 = ',a_temp[0],a_temp[1]
            old_xyz = xyz_temp[0]
            new_xyz,success = self.linear_interpol_structures(xyz_temp,a_temp)
            self.atoms.set_constraint(self.only_other_constraints)
            self.atoms.set_positions(new_xyz)
            self.atoms.set_constraint(self.constraintlist)
            if success:
              print '...success'
            else:
              self.atoms.set_positions(self.scanned_constraint.reset_a(self.a,self.atoms))
            break
          elif (not self.linear_interpol) or (len(a_s)==1 and len(a_temp) == 1) :
            print 'starting structure for a = ',self.a,'is generated from a = ',a_temp[0]
            old_xyz = xyz_temp[0]
            self.atoms.set_constraint(self.only_other_constraints)
            self.atoms.set_positions(xyz_temp[0])
            self.atoms.set_constraint(self.constraintlist) 
            self.atoms.set_positions(self.scanned_constraint.reset_a(self.a,self.atoms))
            print '...success'
            break
 
        new_xyz = self.atoms.get_positions()
        return self.xyz_stepsize_control(new_xyz,old_xyz)
    

    def do_constrained_optimization(self):
        logfile   =self.log_prefix +'relax_'+str(self.a)+'.log'
        trajectory=self.traj_prefix+'relax_'+str(self.a)+'.traj'

        if self.adaptive_threshold == None:
          dyn = BFGS(self.atoms, logfile=logfile, trajectory=trajectory,maxstep=self.maxstep)
          dyn.run(fmax=self.fmax * self.fixed_conv_ratio)
        else:
          dyn = scan_bfgs(self.atoms, logfile=logfile, trajectory=trajectory,maxstep=self.maxstep)
          dyn.pass_constraint(self.scanned_constraint,totfmax=self.fmax)
          dyn.run(fmax=self.adaptive_threshold)

        pg=self.scanned_constraint.projected_force

        self.atoms.set_constraint(self.only_other_constraints)
        e = self.atoms.get_potential_energy()
        f = self.atoms.get_forces()
        fi_max = max([np.linalg.norm(f_1+f_2) for f_1,f_2 in zip (f,self.scanned_constraint.projected_forces)])
        self.atoms.set_constraint(self.constraintlist)
        
  
        if self.a not in self.dopt:
          self.dopt[self.a]={}
        self.dopt[self.a][fi_max]={}
        self.dopt[self.a][fi_max]['pf']=pg
        self.dopt[self.a][fi_max]['e']=e
        self.dopt[self.a][fi_max]['xyz']=self.atoms.get_positions()
        self.dopt[self.a][fi_max]['na_step']=self.na_step
        self.write_restart()

        return e,pg,fi_max,f


