#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
#import lambdify, exp, Symbol, parse_expr, diff
from sympy import lambdify, exp, Symbol, diff, pprint
from sympy.parsing.sympy_parser import parse_expr
from ImportData import *

class Maillage(ImportData) :
    # Classe de maillage initialisée via un fichier à importer
    def __init__(self,data_file,pf=2.,pg=3.) :
        ImportData.__init__(self,data_file)
        
        if vars(self).has_key('Mail') :
            self.Mail.pf=pf
            self.Mail.pg=pg
        elif vars(self).has_key('Geom') :
            self.Geom.typeGeom=typeGeom

        if vars(self).has_key('Func') and self.Geom.typeGeom=='Polar' :
            dfsymb=diff(self.Func.fsymb,'x')
            df2symb=diff(self.Func.fsymb,'x',2)
            Csymb=(np.pi*self.Func.fsymb)**(-2)
            C3symb=-df2symb*dfsymb**(-3)
            C1symb=(self.Func.fsymb*dfsymb)**(-1)+C3symb
            C2symb=dfsymb**(-2)
            f=lambdify(('p','x'),self.Func.fsymb,"numpy")
            self.Func.f=lambda x: f(self.Mail.pf,x)
            C=lambdify(('p','x'),Csymb,"numpy")
            self.Func.C=lambda x: C(self.Mail.pf,x)
            C1=lambdify(('p','x'),C1symb,"numpy")
            self.Func.C1=lambda x: C1(self.Mail.pf,x)
            C2=lambdify(('p','x'),C2symb,"numpy")
            self.Func.C2=lambda x: C2(self.Mail.pf,x)
            C3=lambdify(('p','x'),C3symb,"numpy")
            self.Func.C3=lambda x: C3(self.Mail.pf,x)

            dgsymb=diff(self.Func.gsymb,'x')
            dg2symb=diff(self.Func.gsymb,'x',2)            
            D2symb=dgsymb**(-2)
            D1symb=-D2symb*dg2symb            
            g=lambdify(('p','x'),self.Func.gsymb,"numpy")
            self.Func.g=lambda x: g(self.Mail.pg,x)
            D1=lambdify(('p','x'),D1symb,"numpy")
            self.Func.D1=lambda x: D1(self.Mail.pg,x)
            D2=lambdify(('p','x'),D2symb,"numpy")
            self.Func.D2=lambda x: D2(self.Mail.pg,x)
        
            self.Mail.VecGama=np.linspace(0,1,self.Mail.nb_r)
            self.Mail.VecRadius_norm=self.Func.f(self.Mail.VecGama)
            self.Mail.VecRadius=self.Mail.VecRadius_norm*self.Geom.Radius
        
            self.Mail.VecPhi=np.linspace(0,1,self.Mail.nb_angle)
            self.Mail.VecTheta_norm=self.Func.g(self.Mail.VecPhi)
            self.Mail.VecTheta=self.Mail.VecTheta_norm*self.Geom.angle
        elif vars(self).has_key('Func') and self.Geom.typeGeom=='Rectangular' :
            print 'TODO'
    
    def plotMail(self,fignum,data=[],typeplot='Mesh') :
        plt.figure(fignum)
        if typeplot == 'Scatter' :
            print typeplot
        elif typeplot == 'Contour' :
            print typeplot
        elif typeplot == 'Mesh' : 
            for r in self.Mail.VecRayon :
                plt.scatter(r*np.cos(self.Mail.VecTheta), \
                    r*np.sin(self.Mail.VecTheta),c='b')
        
    
if __name__ == '__main__' :
    print "===================="
    print ""
    print "Maillage :"
    Cable=Maillage('Cable.cfg',1e-3,5)
    
    plt.show()
    fsymb = parse_expr('(1-exp(p*x))/(1-exp(p))')
    dfsymb=diff(fsymb,'x')
    df2symb=diff(fsymb,'x',2)    
    Csymb=(np.pi*fsymb)**(-2)
    C3symb=-df2symb*dfsymb**(-3)    
    C1symb=(fsymb*dfsymb)**(-1)+C3symb
    C2symb=dfsymb**(-2)
    
    
    f=np.vectorize(lambdify(('p','x'),fsymb))
    C1=np.vectorize(lambdify(('p','x'),C1symb))
    C2=np.vectorize(lambdify(('p','x'),C2symb))
    C3=np.vectorize(lambdify(('p','x'),C3symb))

    plt.close('all')
    p=5
    nangle=20
    phi=np.linspace(0,1,nangle)
    Thetaet=f(p,phi)
    Theta=Thetaet*np.pi
    DTheta=np.diff(Theta)
    
    Rayon=1e-2
    ncouronnes=20
    gama=np.linspace(0,1,ncouronnes)
    Ret=f(-p,gama)
    R=Ret*Rayon
    DR=np.diff(R)
    plt.figure()
    plt.axis('equal')
    plt.grid(True)
    monarc=pltp.Arc((0,0),2*Rayon,2*Rayon,theta1=0,theta2=180,color='r')
    plt.gca().add_patch(monarc)
    
    for r in R :
        plt.scatter(r*np.cos(Theta),r*np.sin(Theta),c='b')
        #~ monarc=pltp.Arc((0,0),2*r,2*r,theta1=0,theta2=180)
        #~ plt.gca().add_patch(monarc)
    
    # points au milieu
    for r,dr in zip(R[:-1],DR) :
        monarcmaille=pltp.Arc((0,0),2*(r+dr/2),2*(r+dr/2),theta1=0,theta2=180,color='r')
        plt.gca().add_patch(monarcmaille)
    
    for ang,dt in zip(Theta[0:-1],DTheta) :
        x=np.array([R[1]-DR[0]/2,Rayon])*np.cos(ang+dt/2)
        y=np.array([R[1]-DR[0]/2,Rayon])*np.sin(ang+dt/2)
        plt.plot(x,y,'r-')
    
    # points en conservants les DR et DTheta
    #~ DTheta2=np.copy(DTheta)
    #~ DTheta2[0]=DTheta[0]/2
    #~ Theta2=DTheta2.cumsum()
    
    #~ for r,dr in zip(R[:-1],DR) :
        #~ monarcmaille=pltp.Arc((0,0),2*(r+dr/2),2*(r+dr/2),theta1=0,theta2=180,color='g')
        #~ plt.gca().add_patch(monarcmaille)
    
    #~ for ang in Theta2 :
        #~ x=np.array([R[1]-DR[0]/2,Rayon])*np.cos(ang)
        #~ y=np.array([R[1]-DR[0]/2,Rayon])*np.sin(ang)
        #~ plt.plot(x,y,'g-')

