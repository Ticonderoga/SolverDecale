#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
#import lambdify, exp, Symbol, parse_expr, diff
from sympy import lambdify, exp, Symbol, diff, pprint
from sympy.parsing.sympy_parser import parse_expr
from ImportData import *


if __name__ == '__main__' :
    print "===================="
    print "Tests basiques"
    print "CuMg :"
    Cable=ImportData('Cable.cfg')
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

