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
            
            self.Geom.perimeter=2*self.Geom.height+2*self.Geom.depth
            self.Geom.section=self.Geom.height*self.Geom.depth
            self.Geom.surface=self.Geom.perimeter*2*self.Geom.half_width+2*self.Geom.section
            self.Geom.volume=self.Geom.section*2*self.Geom.half_width
            
def frexp10(x):
    exp = np.floor(np.log10(x))
    return x / 10**exp, exp

def contourtime(time,T,index_time,Bar) :
        
    for i,ind_time in enumerate(index_time) :
        xreal=2*Ls*Bar.Mail.val_f-Ls+pos[ind_time]
        indx=np.where((xreal>=-Bar.Geom.half_width) & (xreal<=Bar.Geom.half_width))
        if 'X' in locals():
            ST=np.r_[ST,T[indx,i].flatten()]
            X=np.r_[X,xreal[indx].flatten()]
            Time=np.r_[Time,time[i]*np.ones(np.size(indx[0]))]
        else :
            ST=T[indx,i].flatten()
            X=xreal[indx].flatten()   
            Time=time[i]*np.ones(np.size(indx[0]))
    
    return Time,X,ST 
    
def mergetime(time1,time2) :
    index=[]
    newtime=np.array([])
    j=0
    if time1[-1]<time2[-1]:
        time1,time2=time2,time1
    for i,t in enumerate(time1[:-1]) :
#        print(i,t,j,time2[j])
        if time2[j]==time1[i]:
            newtime=np.r_[newtime,time2[j]]
            if j<time2.size-1 :
                j=j+1
            index.append(newtime.size-1)
        elif (time2[j]>time1[i]) and (time2[j]<time1[i+1]) :
            newtime=np.r_[newtime,time1[i],time2[j]]           
            if j<time2.size-1 :
                j=j+1 
            index.append(newtime.size-1)
        else :
            newtime=np.r_[newtime,time1[i]]
    newtime=np.r_[newtime,time1[-1]]
    return newtime,np.array(index)
    
if __name__  ==  '__main__' :
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    
    plt.close('all')
    print(30*"=")
    print("")
    print("Maillage :")
    plt.close('all')
    Panto = Maillage('Pantographe.cfg')
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
    Ls=2.
#    Betamax=k/rho/Cp/4/Ls**2*Panto.Func.D1(0.5)+1/2./Ls*vb*Panto.Func.D0(0.5)    
#    dgamacrit=2*k/rho/Cp/Betamax
#    f,p=frexp10(dgamacrit)
##    if f>2 :
##        dgama=np.floor(f-1)*10**p
##    else :
#    dgama=np.floor(f)*10**p
#        
#    n=int(1/dgama)
#    if n%2==0:
#        n=n+1
    n=5001
    Panto.Mail.gama=np.linspace(0,1,n)
    Panto.Mail.val_f=Panto.Func.f(Panto.Mail.gama)
    Panto.Mail.val_D1=Panto.Func.D1(Panto.Mail.gama)            
    Panto.Mail.val_D2=Panto.Func.D2(Panto.Mail.gama)
    Panto.Mail.val_D0=Panto.Func.D0(Panto.Mail.gama)
    
    plt.plot(Panto.Mail.gama,Panto.Mail.val_f,'ob-')
    plt.legend()
    plt.grid()
    
    dgama=Panto.Mail.gama[1]
    Pe1=dgama*vb/k*rho*Cp
    print('Peclet number (vb) : '+str(Pe1))
    
    Pe2=dgama*Betamax/k*rho*Cp
        
    print('Peclet number (Beta) : '+str(Pe2))
    
    dtcrit=dgama/Betamax
    f,p=frexp10(dtcrit)
    dt=10**p
    dt=1000*dt


    sig=Panto.Geom.contact/6
    Period=4*Panto.Geom.sweeping/vb
    
    tf=5*Period
    timebase=np.arange(0,tf,dt)
    timesup=np.arange(Period/4,tf,Period/2)
    time,indxtime_reb=mergetime(timebase,timesup)
    dt_v=np.diff(time)
    h=130.
    
    R=10e-3
    vb_v=vb*scsi.square((time-Period/4)*2*np.pi/Period)
    pos=Panto.Geom.sweeping*scsi.sawtooth((time-Period/4)*2*np.pi/Period,width=0.5)
    Fo=k*dt_v/rho/Cp/dgama**2
    plt.figure(3)

    Tinit=np.zeros_like(Panto.Mail.gama)
    T=Tinit
    I=700.
    schaff_coef=0.38
    surf_coef=1./Panto.Geom.section
    rhs=surf_coef*schaff_coef*np.ones_like(Panto.Mail.gama)/rho/Cp*R*I**2
    source=norm(loc=0.5,scale=sig/2/Ls)
    rhs=rhs/2/Ls*source.pdf(Panto.Mail.val_f)
    savet=1.
    indxtime_reg_save=np.where(time%savet==0)[0]
    indxtime=np.sort(np.unique(np.r_[indxtime_reg_save-1,indxtime_reb-1]))
    indxtime[0]=0
    SaveT=np.empty((n,indxtime.size))
    SaveT[:,0]=Tinit
    savetime=np.empty((indxtime.size,))
    savetime[0]=0
    rhs[0]=0
    rhs[-1]=0
      
    j=1
       
    for i,t in enumerate(time[:-1]) :
        Beta=k/rho/Cp/4./Ls**2*Panto.Mail.val_D1+1/2./Ls*vb_v[i]*Panto.Mail.val_D0

        
        b=-Panto.Mail.val_D2*Fo[i]/4./Ls**2+Beta*dt_v[i]/2./dgama
        a=1+2*Panto.Mail.val_D2*Fo[i]/4./Ls**2+h*Panto.Geom.perimeter*dt_v[i]/rho/Cp/Panto.Geom.section
        c=-Panto.Mail.val_D2*Fo[i]/4./Ls**2-Beta*dt_v[i]/2./dgama
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
            print("time : "+str(time[i+1])+" / Tmin : "+str(min(T))+" / Tmax : "+str(max(T)))
            print(20*"_")
            SaveT[:,j]=T
            savetime[j]=time[i+1]

            if j<indxtime.size-1:
                j=j+1
            
            
#   if ((p+1)%int(interval_savet/dt))==0 :
#            if Affiche_Calcul :
#                print "======================"
#                print "Iteration %d - Temps %.5g" %(p,(p+1)*dt)
#                print "Itération Ray %d - Residu Rayonnement %g - \
#                TJonction=%g °C" %(niter,residu_Ray,Tnew[nb1]-273.15)          
    
    plt.figure(4)
    Tmoy=SaveT.mean(axis=0)
    coef_a=h*Panto.Geom.surface/rho/Cp/Panto.Geom.volume
    coef_b=schaff_coef*R*I**2/rho/Cp/Panto.Geom.volume
    
    Tmoy_anal=coef_b/coef_a*(1-np.exp(-coef_a*savetime))
    plt.plot(savetime,Tmoy_anal,'b',label='analytique')    
    plt.plot(savetime,Tmoy,'r',label='numerique')
    plt.legend()
    plt.grid()
    
    plt.figure(5)
    for j,i in enumerate(indxtime[1:]+1) :
        x=2*Ls*Panto.Mail.val_f-Ls+pos[i]
        ind=np.where((x>=-Panto.Geom.half_width) & (x<=Panto.Geom.half_width))
        plt.plot(x[ind],SaveT[ind,j+1].flatten(),'-',label=str(time[i]))
    plt.legend()
    plt.grid()
