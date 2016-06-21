#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
import matplotlib.tri as tri
#import lambdify, exp, Symbol, parse_expr, diff
from sympy import lambdify, exp, Symbol, diff, pprint
from sympy.parsing.sympy_parser import parse_expr
from ImportDatap3 import *
from Routines.proprietes_fluides import C2K, K2C
from Routines.AnToolsPyxp3 import *
#from Routines.Heat_Transfer.analytical.funcTheta import theta as thetaAnal
import time as tcpu
import scipy.interpolate as scint
import scipy.sparse as scsp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from scipy.stats import norm
import scipy.signal as scsi

class Maillage(ImportData) :
    # Classe de maillage initialisée via un fichier à importer
    def __init__(self, data_file) :
        ImportData.__init__(self,data_file)
        
        if 'Func' in vars(self)  and self.Geom.typeGeom == 'Polar' :

            dfsymb = diff(self.Func.fsymb,'x')
            df2symb = diff(self.Func.fsymb,'x',2)
            Csymb = (np.pi*self.Func.fsymb)**(-2)
            C3symb = -df2symb*dfsymb**(-3)
            C1symb = (self.Func.fsymb*dfsymb)**(-1)+C3symb
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
            D2symb = dgsymb**(-2)
            D1symb = -D2symb*dg2symb            
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
            D1symb = -D2symb*df2symb            
            f = lambdify(('p','x'),self.Func.fsymb,"numpy")
            self.Func.f = lambda x: f(self.Mail.pf,x)
            D1 = lambdify(('p','x'),D1symb,"numpy")
            self.Func.D1 = lambda x: D1(self.Mail.pf,x)
            D2 = lambdify(('p','x'),D2symb,"numpy")
            self.Func.D2 = lambda x: D2(self.Mail.pf,x)            
            D0 = lambdify(('p','x'),D0symb,"numpy")
            self.Func.D0 = lambda x: D0(self.Mail.pf,x)            
    
def frexp10(x):
    exp = floor(math.log10(x))
    return x / 10**exp, exp
    
if __name__  ==  '__main__' :
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    tinit_cpu=tcpu.time()
    plt.close('all')
    print(30*"=")
    print("")
    print("Maillage :")
    plt.close('all')
    Panto = Maillage('Pantographe.cfg')
    gama=np.linspace(0,1,1001)
    ksi=Panto.Func.f(gama)
    plt.plot(gama,ksi,'ob-')
    plt.legend()
    plt.grid()
    Carbon=ImportData('Carbon.cfg')
    vt=140000/3600.
    vb=vt/577.27
    k=Carbon.Ther.conductivity
    rho=Carbon.Ther.density
    Cp=Carbon.Ther.heat_capacity
#    plt.figure(2)
#    for Ls in np.linspace(2,10,9) :
#        Beta=k/rho/Cp/4/Ls**2*Panto.Func.D1(gama)+1/2./Ls*vb*Panto.Func.D0(gama)
#        plt.plot(gama,Beta,'--',label="Ls = "+str(Ls))
    
#    plt.legend()
#    plt.grid()
    dgama=gama[1]
    Pe1=dgama*vb/k*rho*Cp
    print('Peclet number (vb) : '+str(Pe1))
    Ls=5.
    Betamax=k/rho/Cp/4/Ls**2*Panto.Func.D1(0.5)+1/2./Ls*vb*Panto.Func.D0(0.5)
    Pe2=dgama*Betamax/k*rho*Cp
        
    print('Peclet number (Beta) : '+str(Pe2))
    
    dtcrit=dgama/Betamax
    f,p=frexp10(dtcrit)
    dt=10**(-5)
    
    LB=766e-3
    Lb=400e-3
    Lc=2e-3
    sig=Lc/6
    Period=2*Lb/vb
    

    time=np.arange(0,2*Period,dt)
    h=130.
    Pcv=2*50e-3+2*32e-3
    S=50e-3*32e-3
    R=Carbon.Elec.resistivity*LB/S
    vb_v=vb*scsi.square((time-Period/2)*2*np.pi/Period)
    Fo=k*dt/rho/Cp/dgama**2
    plt.figure(3)

    Tinit=np.zeros_like(gama)
    T=Tinit
    I=1000.
    rhs=np.ones_like(gama)*dt/rho/Cp*R*I**2/LB/S
    source=norm(loc=0.5,scale=sig/2/Ls)
    rhs=rhs*1/2./Ls*source.pdf(gama)
    SaveT=[]
    SaveT.append(Tinit)
    rhs[0]=0
    rhs[-1]=0
    
    for i,t in enumerate(time) :
        if vb_v[i+1]!=vb_v[i] :
            print(t)
            Beta=Betap=Betam=np.zeros_like(gama)
        else:
            Beta=k/rho/Cp/4/Ls**2*Panto.Func.D1(gama)+1/2./Ls*vb_v[i]*Panto.Func.D0(gama)
            Betap=np.maximum(Beta,np.zeros_like(Beta))
            Betam=np.minimum(Beta,np.zeros_like(Beta))
            
        b=-Panto.Func.D2(gama)*Fo/4/Ls**2+Betap*dt/dgama
        a=1+2*Panto.Func.D2(gama)*Fo/4/Ls**2+dt/dgama*(Betam-Betap)+h*Pcv*dt/rho/Cp/S
        c=-Panto.Func.D2(gama)*Fo/4/Ls**2-Betam*dt/dgama
        a[0]=1
        a[-1]=1
        b=b[1:]
        b[-1]=0
        c=c[:-1]
        c[0]=0
        
#    for t in time :
        T=Solvetridiag(b,a,c,T+rhs)
#        if min(T)<0 :
#            break
        if i%1000 ==0 :
            print("time : "+str(t)+" / Tmin : "+str(min(T))+" / Tmax : "+str(max(T)))
            x=2*Ls*Panto.Func.f(gama)-Ls+vb_v[i]*t
            ind=np.where((x>=-LB) & (x<=LB))
            plt.plot(x[ind],T[ind],'-',label=str(t))
    
    plt.legend()
    plt.grid()