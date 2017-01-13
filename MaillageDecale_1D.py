#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Utilisation de tri pour le Maillage 2D polaire
import matplotlib.tri as tri
# fonctions de calcul symbolique utilisée dans Maillage
from sympy import lambdify, exp, Symbol, diff
from sympy.parsing.sympy_parser import parse_expr
# La classe Maillage pourrait se retrouver dans ImportData3
from ImportDatap3 import *

# Résolution tridiag
from Routines.AnToolsPyxp3 import *

import scipy.stats as scst
import scipy.signal as scsi


class Maillage(ImportData) :
    """ Classe de maillage initialisée via un fichier à importer
    on peut gérer un maillage polaire 2D ou 1D
    l'idée générale est d'avoir des fonctions puis des valeurs sur
    le maillage en Gama """
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
    """Fonction permettant de fusionner deux bases de temps 
    et surtout de donner les indices de time2 dans le nouveau
    vecteur de temps
    """
    tmerge=np.r_[time1,time2]
    Ind=np.argsort(tmerge)
    indices=np.where(Ind>time1.size)[0]
    return tmerge[Ind],indices

def mergetime2(time1,time2,ratio) :
    """Fonction permettant de fusionner deux bases de temps 
    ratio représente le ratio (entier) entre savet et dt
    
    Exemple : 
    --------
    
    >>>  ratio = int(np.floor(savet/dt))
    >>>  time1 = np.arange(0,tf,dt)
    >>>  time2 = # points supplémentaires
    >>>  time,ind = mergetime2(time1,time2,ratio)
     
    Returns : 
    --------
    **time** : la base de temps fusionnée
    
    **ind** : les indices de **time2** et des temps tous les **savet** dans **time**
     """
    tmerge,Ind,counts=np.unique(np.r_[time1,time2],return_index=True,return_counts=True)
    if counts.sum()!=(time1.size+time2.size):
        print('Warning bases de temps se chevauchent')                
    indices_ratio=np.where((Ind%ratio==0)&(Ind<time1.size))[0]
    indices_time2=np.where(Ind>=time1.size)[0]
    indices = np.union1d(indices_ratio,indices_time2)
 
    return tmerge,indices,indices_time2

if __name__  ==  '__main__' :
    
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    
    plt.close('all')

    # Pantographe regroupe les infos du Maillage sur le Panto
    # Il y a donc la géométrie mais aussi le Maillage    
    Panto = Maillage('Pantographe.cfg')    
    # le matériau : Ici du Carbone
    Carbon=ImportData('Carbon.cfg')
    # le trajet pour l'instant c'est là qu'on met le courant
    Case = ImportData('ConfigTrip.cfg')
    
    # vitesse de balayge
    # TOFIX il faudrait que la vitesse de balayage soit fonction
    # de la vitesse du train mais aussi du plan de piquetage
    # on peut envisager de mettre des .dat dans le fichier ConfigTrip
    sweeping_velocity = Case.Trai.velocity / 577.27
    
    # par facilité mais il faudrait l'enlever 
    k=Carbon.Ther.conductivity
    rho=Carbon.Ther.density
    Cp=Carbon.Ther.heat_capacity

    # le contact de la caténaire couvre 6 sigma de la gaussienne
    sig=Panto.Geom.contact/6
    
    # On calcule la pente de la fonction de raffinage pour en déduire un
    # nombre de points au contact
    pente_f_center=lambdify(('p','x'),diff(Panto.Func.fsymb,'x'))(Panto.Mail.pf,0.5)
    nb_center=np.floor(3*sig/Panto.Mail.Ls/pente_f_center/Panto.Mail.dGama)
    print('Il y a %d points au contact'%nb_center)
    
    # On calcule la période de temps
    Period=4*Panto.Geom.sweeping/sweeping_velocity
    # les points temporels classiques
    timebase=np.arange(0,Case.Time.final,Case.Time.dt)
    # les points de rebroussement
    timesup=np.arange(Period/4,Case.Time.final,Period/2)
    ratio=int(np.floor(Case.Time.savet/Case.Time.dt))
    time,indxtime,indx_reb=mergetime2(timebase,timesup,ratio)
    # les dt sous forme vectorielle la plupart sont égaux à Case.Time.dt
    dt_v=np.diff(time)
    
    # Coefficient d'échange en W/m2/K il faudrait le recalculer à partir
    # de la vitesse
    h=130.
    
    # les vitesses de balayage en vectoriel
    sweeping_velocity_v=sweeping_velocity*scsi.square((time-Period/4)*2*np.pi/Period)
    
    # les positions de la caténaire
    pos=Panto.Geom.sweeping*scsi.sawtooth((time-Period/4)*2*np.pi/Period,width=0.5)
    # le nombre de Fourier
    Fo=k*dt_v/rho/Cp/Panto.Mail.dGama**2
    
    # Température initiale 
    # attention on calcule un échauffement et Tinit=0
    Tinit=np.zeros_like(Panto.Mail.Gama)
    T=Tinit
    
    # le terme d'échauffement lié au contact surfacique
    rhs_constant=Carbon.Elec.schaff_coef/Panto.Geom.section\
        /rho/Cp*Carbon.Elec.contact*Case.Elec.current**2
    # la gaussienne qu'il faudra diviser par 2Ls
    source=scst.norm(loc=0.5,scale=sig/2/Panto.Mail.Ls)
    # le terme source final
    rhs = rhs_constant*source.pdf(Panto.Mail.val_f)/2/Panto.Mail.Ls
    # au bout rien ..
    rhs[0]=0
    rhs[-1]=0
    
    # les indices de temps pour la sauvegarde
#    indxtime[0]=0

    # initialisation du  tableau des températures
    SaveT=np.empty((Panto.Mail.n,indxtime.size))
    SaveT[:,0]=Tinit
    # initialisation des temps de sauvegarde (Redondance ?)
    savetime=np.empty((indxtime.size,))
    savetime[0]=0
    Tmid=np.empty((indxtime.size))
    Tmid[0]=Tinit[0]
    #%% debut boucle temporelle
    j=1
    k=0   
    for i,t in enumerate(time[:-1]) :

        if i==indx_reb[k] and k<indx_reb.size-1:
            sweeping_velocity_v[i]=-sweeping_velocity_v[i-1]
            k=k+1
            
        # Coefficient sur l'advection
        Beta=k/rho/Cp/4./Panto.Mail.Ls**2*Panto.Mail.val_D1\
            +1/2./Panto.Mail.Ls*sweeping_velocity_v[i]*Panto.Mail.val_D0
        # les 3 diagonales b,a,c
        b=-Panto.Mail.val_D2*Fo[i]/4./Panto.Mail.Ls**2+Beta*dt_v[i]/2./Panto.Mail.dGama

        a=1+2*Panto.Mail.val_D2*Fo[i]/4./Panto.Mail.Ls**2+\
            h*Panto.Geom.perimeter*dt_v[i]/rho/Cp/Panto.Geom.section
        
        c=-Panto.Mail.val_D2*Fo[i]/4./Panto.Mail.Ls**2-Beta*dt_v[i]/2./Panto.Mail.dGama
        
        # les conditions limites simplissimes
        a[0]=1.
        a[-1]=1.
        b=b[1:]
        b[-1]=0.
        c=c[:-1]
        c[0]=0.
        
        # la résolution du système tridiagonal
        T=Solvetridiag(b,a,c,T+rhs*dt_v[i])

        # on va afficher et sauvegarder les valeurs
        if (i+1)==indxtime[j] :
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

    # on trace            
    plt.figure(2)
    for j,i in enumerate(indxtime) :
        # recalcul des vraies positions
        x=2*Panto.Mail.Ls*Panto.Mail.val_f-Panto.Mail.Ls+pos[i]
        ind=np.where((x>=-Panto.Geom.half_width) & (x<=Panto.Geom.half_width))
        plt.plot(x[ind],SaveT[ind,j].flatten(),'-')
        Tmid[j]=np.interp(-0.1, x[ind],SaveT[ind,j].flatten())


    plt.grid()
    plt.title('Num : Ls = '+str(Panto.Mail.Ls)+' : n = '+str(Panto.Mail.n)+\
        ' : dt = '+str(Case.Time.dt)+' : p = '+str(Panto.Mail.pf))
    plt.xlabel('distance [m]')
    plt.ylabel('Echauffement [°C]')
    # on règles les axes
    current_axis=plt.axis()
    plt.axis([current_axis[0],current_axis[1],0,current_axis[3]])
    # sauvegarde
    plt.savefig("./Results/Num_"+str(Panto.Mail.Ls)+"_"+str(Panto.Mail.n)+\
        "_"+str(Case.Time.dt)+"_"+str(Panto.Mail.pf)+".pdf")
    
    plt.figure(3)
    plt.plot(savetime,Tmid,'b*-')
    plt.grid()
    plt.xlabel('Temps [s]')
    plt.ylabel('Echauffement [°C]')
    plt.title('Température vs. time at x=-0.1 m')



# BOUT DE CODE TRASH pour sauvegarde des Résultats    
#
#    np.savez_compressed("./Results/Data_"+str(Panto.Mail.Ls)+"_"+str(n),\
#        indxtime=indxtime,\
#        val_f=Panto.Mail.val_f,\
#        pos=pos,\
#        SaveT=SaveT,\
#        savetime=savetime)

