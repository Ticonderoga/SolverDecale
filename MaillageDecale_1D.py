#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.tri as tri
from sympy import lambdify, exp, Symbol, diff
from sympy.parsing.sympy_parser import parse_expr
from ImportDatap3 import *
from Routines.proprietes_fluides import C2K, K2C
from Routines.AnToolsPyxp3 import *

import scipy.interpolate as scint
import scipy.sparse as scsp

#from concurrent.futures import ThreadPoolExecutor
#import multiprocessing

import scipy.stats as scst
import scipy.signal as scsi
import sys

class Maillage(ImportData) :
    # Classe de maillage initialisée via un fichier à importer
    def __init__(self, data_file) :
        ImportData.__init__(self,data_file)
        
        if 'Func' in vars(self)  and self.Geom.typeGeom == 'Polar' :

            dfsymb = diff(self.Func.fsymb,'x')
            df2symb = diff(self.Func.fsymb,'x',2)
            Csymb = (np.pi*self.Func.fsymb)**(-2)#ok
            C3symb = -df2symb*dfsymb**(-3)#ok
            C1symb = (self.Func.fsymb*dfsymb)**(-1)+C3symb#ok
            C2symb = dfsymb**(-2)
            f = lambdify(('p','x'),self.Func.fsymb,"numpy")
            self.Func.f = lambda x: f(self.Mail.pf,x)
            C = lambdify(('p','x'),Csymb,"numpy")
            self.Func.C = lambda x: C(self.Mail.pf,x)

            C1 = lambdify(('p','x'),C1symb,"numpy")
            self.Func.C1 = lambda x: C1(self.Mail.pf,x)
            C2 = lambdify(('p','x'),C2symb,"numpy")
            self.Func.C2 = lambda x: C2(self.Mail.pf,x)

            
            C3 = lambdify(('p','x'),C3symb,"numpy")
            self.Func.C3 = lambda x: C3(self.Mail.pf,x)


            dgsymb = diff(self.Func.gsymb,'x')
            dg2symb = diff(self.Func.gsymb,'x',2)
            D2symb = dgsymb**(-2)#ok
            D1symb = -dg2symb*dgsymb**(-3)#ok      
            g = lambdify(('p','x'),self.Func.gsymb,"numpy")
            self.Func.g = lambda x: g(self.Mail.pg,x)
            D1 = lambdify(('p','x'),D1symb,"numpy")
            self.Func.D1 = lambda x: D1(self.Mail.pg,x)
            D2 = lambdify(('p','x'),D2symb,"numpy")
            self.Func.D2 = lambda x: D2(self.Mail.pg,x)
        
            self.Mail.Gama = np.linspace(0,1,self.Mail.nb_r)
            self.Mail.val_C=self.Func.C(self.Mail.Gama)
            self.Mail.val_C1=self.Func.C1(self.Mail.Gama)            
            self.Mail.val_C2=self.Func.C2(self.Mail.Gama)
            self.Mail.val_C3=self.Func.C3(self.Mail.Gama)
            
            
            self.Mail.val_f = self.Func.f(self.Mail.Gama)
            self.Mail.Radius = self.Mail.val_f*self.Geom.radius
            self.Mail.diffRadius = np.diff(self.Mail.Radius)
        
            self.Mail.Phi = np.linspace(0,1,self.Mail.nb_angle)
            
            
            self.Mail.val_g = self.Func.g(self.Mail.Phi)
            self.Mail.Theta = self.Mail.val_g*self.Geom.angle
            self.Mail.diffTheta = np.diff(self.Mail.Theta)
            
            self.Mail.grid_R,self.Mail.grid_Theta = \
                np.meshgrid(self.Mail.Radius,self.Mail.Theta)
            self.trigint_R,self.trigint_Theta = np.meshgrid( \
                np.linspace(0,self.Geom.radius,100), \
                np.linspace(0,self.Geom.angle,100))
            self.trigint_R,self.trigint_Theta = \
                self.trigint_R.flatten(), \
                self.trigint_Theta.flatten()
            
            self.Mail.triang=tri.Triangulation( \
                self.trigint_R*np.cos(self.trigint_Theta), \
                self.trigint_R*np.sin(self.trigint_Theta))

        elif 'Func' in vars(self) and self.Geom.typeGeom == '1D' :
            dfsymb = diff(self.Func.fsymb,'x')
            df2symb = diff(self.Func.fsymb,'x',2)
            D2symb = dfsymb**(-2)
            D0symb = dfsymb**(-1)

            D1symb = -df2symb*dfsymb**(-3)
            f = lambdify(('p','x'),self.Func.fsymb,"numpy")
            self.Func.f = lambda x: f(self.Mail.pf,x)
            D1 = lambdify(('p','x'),D1symb,"numpy")
            self.Func.D1 = lambda x: D1(self.Mail.pf,x)
            D2 = lambdify(('p','x'),D2symb,"numpy")
            self.Func.D2 = lambda x: D2(self.Mail.pf,x)            
            D0 = lambdify(('p','x'),D0symb,"numpy")
            self.Func.D0 = lambda x: D0(self.Mail.pf,x)
            self.Mail.n = int(self.Mail.n)
            self.Mail.Gama = np.linspace(0,1,self.Mail.n)
            self.Mail.dGama = self.Mail.Gama[1] 
            
            self.Mail.val_f=self.Func.f(self.Mail.Gama)
            self.Mail.val_D1=self.Func.D1(self.Mail.Gama)            
            self.Mail.val_D2=self.Func.D2(self.Mail.Gama)
            self.Mail.val_D0=self.Func.D0(self.Mail.Gama)
    
            
            self.Geom.perimeter=2*self.Geom.height+2*self.Geom.depth
            self.Geom.section=self.Geom.height*self.Geom.depth
            self.Geom.surface=self.Geom.perimeter*2*self.Geom.half_width+2*self.Geom.section
            self.Geom.volume=self.Geom.section*2*self.Geom.half_width
            
    
def mergetime(time1,time2) :
    tmerge=np.r_[time1,time2]
    Ind=np.argsort(tmerge)
    indices=np.where(Ind>time1.size)[0]
    return tmerge[Ind],indices


if __name__  ==  '__main__' :
    
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    
    plt.close('all')
    print(30*"=")
    print("")
    print("Maillage :")
    
    Panto = Maillage('Pantographe.cfg')
    
    Carbon=ImportData('Carbon.cfg')
    Case = ImportData('ConfigTrip.cfg')

    # sweeping velocity should be detailed 
    # especially the 577.27    
    sweeping_velocity = Case.Trai.velocity / 577.27
    
    k=Carbon.Ther.conductivity
    rho=Carbon.Ther.density
    Cp=Carbon.Ther.heat_capacity
    Ls=10.

    sig=Panto.Geom.contact/6
    Period=4*Panto.Geom.sweeping/sweeping_velocity
    
    timebase=np.arange(0,Case.Time.final,Case.Time.dt)
    timesup=np.arange(Period/4,Case.Time.final,Period/2)    
    time,indxtime_reb=mergetime(timebase,timesup)
    dt_v=np.diff(time)
    h=130.
    
    
    sweeping_velocity_v=sweeping_velocity*scsi.square((time-Period/4)*2*np.pi/Period)
    pos=Panto.Geom.sweeping*scsi.sawtooth((time-Period/4)*2*np.pi/Period,width=0.5)
    Fo=k*dt_v/rho/Cp/Panto.Mail.dGama**2
    

    Tinit=np.zeros_like(Panto.Mail.Gama)
    T=Tinit
    
    rhs_constant=Carbon.Elec.schaff_coef/Panto.Geom.section\
        /rho/Cp*Carbon.Elec.contact*Case.Elec.current**2
    source=scst.norm(loc=0.5,scale=sig/2/Ls)

    rhs = rhs_constant*source.pdf(Panto.Mail.val_f)/2/Ls

    rhs[0]=0
    rhs[-1]=0
    
    indxtime_reg_save=np.where(time%Case.Time.savet==0)[0]
    indxtime=np.sort(np.unique(np.r_[indxtime_reg_save-1,indxtime_reb-1]))
    indxtime[0]=0
    SaveT=np.empty((Panto.Mail.n,indxtime.size))
    SaveT[:,0]=Tinit
    savetime=np.empty((indxtime.size,))
    savetime[0]=0
    
    #%% debut boucle temporelle
    j=1
       
    for i,t in enumerate(time[:-1]) :
        Beta=k/rho/Cp/4./Ls**2*Panto.Mail.val_D1\
            +1/2./Ls*sweeping_velocity_v[i]*Panto.Mail.val_D0
        
        b=-Panto.Mail.val_D2*Fo[i]/4./Ls**2+Beta*dt_v[i]/2./Panto.Mail.dGama

        a=1+2*Panto.Mail.val_D2*Fo[i]/4./Ls**2+\
            h*Panto.Geom.perimeter*dt_v[i]/rho/Cp/Panto.Geom.section
        
        c=-Panto.Mail.val_D2*Fo[i]/4./Ls**2-Beta*dt_v[i]/2./Panto.Mail.dGama
        a[0]=1.
        a[-1]=1.
        b=b[1:]
        b[-1]=0.
        c=c[:-1]
        c[0]=0.
        
        T=Solvetridiag(b,a,c,T+rhs*dt_v[i])

        if i==indxtime[j] :
            print(10*"_")
            print(10*" "+"\\"+(20-1-10)*"_")
            print('i : '+str(i))            
            print("time : "+'{:.2f}'.format(time[i+1])+\
                  "s / Tmin : "+'{:.2f}'.format(min(T))+\
                  "°C / Tmax : "+'{:.2f}'.format(max(T))+' °C')
            print(20*"_")
            SaveT[:,j]=T
            savetime[j]=time[i+1]

            if j<indxtime.size-1:
                j=j+1

    plt.figure(2)
    for j,i in enumerate(indxtime[1:]+1) :
        x=2*Ls*Panto.Mail.val_f-Ls+pos[i]
        ind=np.where((x>=-Panto.Geom.half_width) & (x<=Panto.Geom.half_width))
        plt.plot(x[ind],SaveT[ind,j+1].flatten(),'-')

    plt.legend()
    plt.grid()
    plt.title('Num : Ls = '+str(Ls)+' : n = '+str(Panto.Mail.n)+\
        ' : dt = '+str(Case.Time.dt)+' : p = '+str(Panto.Mail.pf))
    plt.xlabel('distance [m]')
    plt.ylabel('Echauffement [°C]')
    current_axis=plt.axis()
    plt.axis([current_axis[0],current_axis[1],0,current_axis[3]])
    plt.savefig("./Results/Num_"+str(Ls)+"_"+str(Panto.Mail.n)+\
        "_"+str(Case.Time.dt)+"_"+str(Panto.Mail.pf)+".pdf")
    
#    np.savez_compressed("./Results/Data_"+str(Ls)+"_"+str(n),\
#        indxtime=indxtime,\
#        val_f=Panto.Mail.val_f,\
#        pos=pos,\
#        SaveT=SaveT,\
#        savetime=savetime)

#    plt.close('all')
