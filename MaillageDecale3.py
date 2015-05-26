#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
#import lambdify, exp, Symbol, parse_expr, diff
from sympy import lambdify, exp, Symbol, diff, pprint
from sympy.parsing.sympy_parser import parse_expr
from ImportData import *
from Routines.proprietes_fluides import C2K, K2C
from Routines.AnToolsPyx import *
from Routines.Heat_Transfer.analytical.funcTheta import theta


class Maillage(ImportData) :
    # Classe de maillage initialisée via un fichier à importer
    def __init__(self, data_file, pf=2., pg=3.) :
        ImportData.__init__(self,data_file)
        
        if vars(self).has_key('Mail') :
            self.Mail.pf = pf
            self.Mail.pg = pg
        elif vars(self).has_key('Geom') :
            self.Geom.typeGeom = typeGeom

        if vars(self).has_key('Func') and self.Geom.typeGeom == 'Polar' :

#            D=Cable.Func.fsymb.atoms(Symbol) 
#            for s in D:
#                if s.name=='p' :
#                    print "Yes"
#                else :
#                    print "Bad"

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
            self.Mail.val_f = self.Func.f(self.Mail.Gama)
            self.Mail.Radius = self.Mail.val_f*self.Geom.radius
            self.Mail.diffRadius = np.diff(self.Mail.Radius)
        
            self.Mail.Phi = np.linspace(0,1,self.Mail.nb_angle)
            self.Mail.val_g = self.Func.g(self.Mail.Phi)
            self.Mail.Theta = self.Mail.val_g*self.Geom.angle
            self.Mail.diffTheta = np.diff(self.Mail.Theta)
        elif vars(self).has_key('Func') and self.Geom.typeGeom == 'Rectangular' :
            print 'TODO'
    
    def plotData(self,fignum,data = [],typeplot = 'Mesh') :
        # ne fonctionne qu'avec du Polar        
        # on trace ici seulement les points et non les mailles 
        
        plt.figure(fignum)
        plt.axis('equal')
        plt.grid(True)
        if typeplot  ==  'Scatter' :
            plt.scatter(np.outer(Cable.Mail.Radius,np.cos(Cable.Mail.Theta)),\
                np.outer(Cable.Mail.Radius,np.sin(Cable.Mail.Theta)),c = data)
            plt.colorbar()
        elif typeplot  ==  'Contour' :
            print typeplot
        elif typeplot  ==  'Mesh' : 
            for r in self.Mail.Radius :
                plt.scatter(r*np.cos(self.Mail.Theta), \
                    r*np.sin(self.Mail.Theta),c = 'b')

    def plotMail(self,fignum,col = 'k') :
        # ne fonctionne qu'avec du Polar        
        # on trace ici seulement les mailles et non les points
        
        # arc extérieur
        plt.figure(fignum)                
        plt.axis('equal')
        plt.grid(True)
        arc_ext = pltp.Arc((0,0),2*self.Geom.radius,2*self.Geom.radius, \
            theta1 = 0,theta2 = np.rad2deg(self.Geom.angle),color = col,ls = 'dashed')
        plt.gca().add_patch(arc_ext)
        
        # arcs intérieurs
        for r,dr in zip(self.Mail.Radius[:-1],self.Mail.diffRadius) :
            arc_int = pltp.Arc((0,0),2*(r+dr/2),2*(r+dr/2), \
                theta1 = 0,theta2 = np.rad2deg(self.Geom.angle),color = col,ls = 'dashed')
            plt.gca().add_patch(arc_int)
        # rayons
        for ang,dt in zip(self.Mail.Theta[:-1],self.Mail.diffTheta) :
            x = np.array([self.Mail.Radius[1]-self.Mail.diffRadius[0]/2,self.Geom.radius])*np.cos(ang+dt/2)
            y = np.array([self.Mail.Radius[1]-self.Mail.diffRadius[0]/2,self.Geom.radius])*np.sin(ang+dt/2)
            plt.plot(x,y,col+'--')


def split_data(k) :
    """ for a matrix k split_data(k) will return all the values in between
    example :
    [ki,kj] = split_data(k)
    """

    ki = (k[1::,:]+k[:-1:,:])/2.
    kj = (k[:,1::]+k[:,:-1:])/2. 
    return ki,kj


def padding(M) :
    Mi,Mj=split_data(M)
    Mi_mj=np.pad(Mi,((1,0),(0,0)),mode='constant')
    Mi_pj=np.pad(Mi,((0,1),(0,0)),mode='constant')
    Mij_m=np.pad(Mj,((0,0),(1,0)),mode='constant')
    Mij_p=np.pad(Mj,((0,0),(0,1)),mode='constant')
    
    return Mi_mj,Mi_pj,Mij_m,Mij_p

def fake_source(Wire,p=2,Qmax=0.) :
    """Function to fake an inhomogenous heat source in a wire 
    Q=fake_source(Wire,p=3,Qavg=0.) 
    with     
    Wire : Maillage object
    p : parameter to refine near the spot. Default Value 2
    Qavg : Maximal value of Q. Default value 0."""
    R = Wire.Geom.radius
    r = Wire.Mail.Radius
    theta = Wire.Mail.Theta
    
    d = ((np.tile(r,(Wire.Mail.nb_angle,1)).T)**2+R**2-2*R*np.outer(r,np.cos(theta)))**0.5
    
    return Qmax*(1-np.exp(p*(2*R-d)/2/R))/(1-np.exp(p))
    
def mul3cols(Bm,B,Bp,T) :
    n,m=T.shape
    supB=np.array([Bm,B,Bp])
    result=np.empty_like(T)    
    for j in range(1,m-1) :
        M1=supB[:,:,j].T
        M2=T[:,j-1:j+2]
        result[:,j]=(M1*M2).sum(-1)
    
    result[:,0]=(supB[1:,:,0].T*T[:,0:2]).sum(-1)
    result[:,-1]=(supB[:-1,:,-1].T*T[:,-2:]).sum(-1)

    return result
   
def calTcenter(Toldc,T) :
    Tm1=T[1,:].mean()
    Tm2=T[2,:].mean()
    Tcenter=(Toldc+heat_source[0,:]*dt/2/Material.Ther.density/Material.Ther.heat_capacity \
            + Foij[0,:]*(Cable.Func.C3(0)/dGama-2*Cable.Func.C2(0)/dGama**2)*Tm1 \
            + Foij[0,:]*Cable.Func.C2(0)/dGama**2*Tm2) \
            / (1+Foij[0,:]*(Cable.Func.C3(0)/dGama-Cable.Func.C2(0)/dGama**2))
    return Tcenter

    
if __name__  ==  '__main__' :
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    
    plt.close('all')
    print 30*"="
    print ""
    print "Maillage :"
    plt.close('all')
    Cable = Maillage('Cable.cfg',-3.,3.)
    Cable.plotMail(2,'r')
    Cable.plotData(2,1,'Mesh')
    print 30*"="
    print ""
    print "Matériau :"
    Material = ImportData('Cu.cfg')
    idxi = np.arange(int(Cable.Mail.nb_r))
    idxj = np.arange(int(Cable.Mail.nb_angle))
    f = np.r_[Cable.Mail.val_f,1,0]
    g = np.r_[Cable.Mail.val_g,1,0]

    fn_1=Cable.Mail.val_f[-2]
    gm_1=Cable.Mail.val_g[-2]
#    pour se faciliter la tâche
    m=int(Cable.Mail.nb_angle-1)
    n=int(Cable.Mail.nb_r-1)
#    fin
    aj = Cable.Geom.angle/2*(g[idxj+1]-g[idxj-1])
    li = Cable.Geom.radius/2*(f[idxi+1]-f[idxi-1])
    cij_p = Cable.Geom.radius/2*np.outer(f[idxi]+f[idxi+1],aj)
    cij_m = Cable.Geom.radius/2*np.outer(f[idxi-1]+f[idxi],aj)
    sij = Cable.Geom.radius/4*np.outer(li*(f[idxi+1]+2*f[idxi]+f[idxi-1]),aj)
    sij[0,:] = sij[0,:].sum()
    
    
    dGama = Cable.Mail.Gama[1]
    dPhi = Cable.Mail.Phi[1]

#    plt.figure()
#    plt.axis('equal')
#    Cable.plotMail(3,'k')
#    Cable.plotData(3,data = sij,typeplot = 'Scatter')
    Tinit = C2K(100.)
    Tinit_vec = Tinit*np.ones((Cable.Mail.nb_r,Cable.Mail.nb_angle))
    T = Tinit_vec
#    On fait un vecteur T bruité
#    T=T*(1+0.3*np.random.randn(Cable.Mail.nb_r,Cable.Mail.nb_angle))

    k = Material.Ther.conductivity(T)
     
    tf = 1200                # secondes
    nbdt = tf*10+1            # nb points en temps
    time,dt = np.linspace(0,tf,nbdt,retstep = True)

    [ki,kj] = split_data(k)
    Foij = k/Material.Ther.density/Material.Ther.heat_capacity*dt/Cable.Geom.radius**2
    
    Foi_mj,Foi_pj,Foij_m,Foij_p=padding(Foij)

#    TODO Vérifier Calcul de Biot radius ou delta_r
    h = Cable.CL.heat_transfer_coefficient
    Bi = h*Cable.Geom.radius/k[-1,:]
# ____________________    
#/    Calcul alpha    \_______________________________

    temp = Cable.Func.C1(Cable.Mail.Gama)[:,np.newaxis]*Foij/4./dGama
    temp2_m = Cable.Func.C2(Cable.Mail.Gama)[:,np.newaxis]*Foi_mj/2./dGama**2
    temp2_p = Cable.Func.C2(Cable.Mail.Gama)[:,np.newaxis]*Foi_pj/2./dGama**2
    
    alpha_m = temp-temp2_m
    alpha_p = -temp-temp2_p
    alpha = temp2_m + temp2_p
#    TODO Vérifier eta_nj
    eta_nj = 4*Bi*Foij[-1,:]/(1-fn_1)/(3+fn_1)
    alpha_m_nj = -2*Foi_mj[-1,:]*(1+fn_1)/(1-fn_1)**2/(3+fn_1)
    alpha[-1,:] = eta_nj-alpha_m_nj
    alpha[0,:] = 0
    alpha_m[0,:] = np.inf
    alpha_m[-1,:] = alpha_m_nj
    alpha_p[0,:] = 0
    alpha_p[-1,:] = np.inf
    
            
# ___________________
#/    Calcul beta    \_______________________________
#    
    temp = (np.dot(Cable.Func.C(Cable.Mail.Gama)[:,np.newaxis], \
        Cable.Func.D1(Cable.Mail.Phi)[np.newaxis,:])*Foij/4./dPhi)
    temp2_m = np.dot(Cable.Func.C(Cable.Mail.Gama)[:,np.newaxis], \
        Cable.Func.D2(Cable.Mail.Phi)[np.newaxis,:])*Foij_m/2./dPhi**2
    temp2_p = np.dot(Cable.Func.C(Cable.Mail.Gama)[:,np.newaxis], \
        Cable.Func.D2(Cable.Mail.Phi)[np.newaxis,:])*Foij_p/2./dPhi**2

    beta_m = -temp + temp2_m
    beta_p = temp + temp2_p
    beta = beta_m + beta_p

# Beta n,j  Eq 39_________________    
    beta_m_nj = 4*Foij_m[-1,:]/np.pi**2/(3+fn_1)/ \
            ((g[idxj+1]-g[idxj-1])*(g[idxj]-g[idxj-1]))
            
    beta_m[-1,:] = beta_m_nj
    
    beta_p_nj = 4*Foij_p[-1,:]/np.pi**2/(3+fn_1)/ \
            ((g[idxj+1]-g[idxj-1])*(g[idxj+1]-g[idxj]))

    beta_p[-1,:] = beta_p_nj
    beta[-1,:] = beta_m_nj + beta_p_nj

# Beta i,0  Eq 41_________________    
    beta_m[:,0] = 0
    beta_p[:,0] = Foij_p[:,0]*Cable.Func.C(Cable.Mail.Gama)* \
        Cable.Func.D2(Cable.Mail.Phi)[0]/dPhi**2
    beta[:,0] = beta_p[:,0]

# Beta n,0  Eq 42_________________    
    beta_m[-1,0] = 0
    beta_p[-1,0] = 4*Foij_p[-1,0]/np.pi**2/(3+fn_1)/g[1]**2
    beta[-1,0] = beta_p[-1,0]


# Beta i,m  Eq 44_________________    
    beta_m[:,-1] = Foij_m[:,-1]*Cable.Func.C(Cable.Mail.Gama)* \
        Cable.Func.D2(Cable.Mail.Phi)[-1]/dPhi**2
    beta_p[:,-1] = 0
    beta[:,-1] = beta_m[:,-1]
    
# Beta n,m  Eq 45_________________    
    beta_m[-1,-1] = 4*Foij_m[-1,-1]/np.pi**2/(3+fn_1)/(1-gm_1)**2  
    beta_p[-1,-1] = 0
    beta[-1,-1] = beta_m[-1,-1]
    
# Boucle de convergence selon les rayons
    niter=0
    convergence=True
    while convergence :
        heat_source=fake_source(Cable,p=2,Qmax=0.)
        rhs1=mul3cols(beta_m,1-beta,beta_p,T)
        Toldc=T[0,:]
        rhs1[0,:]=Toldc
        rhs2=heat_source*dt/2/Material.Ther.density/Material.Ther.heat_capacity
        rhs2[-1,:]=rhs2[-1,:]+eta_nj*Cable.CL.Tinf
        rhs2[0,:]=0
        rhs=rhs1+rhs2
        T1step=np.empty_like(T)
        for j in range(m+1) :
            T1step[:,j]=Solution_Thomas(alpha_m[1:,j],1+alpha[:,j],alpha_p[:-1,j],rhs[:,j])
    
        Tnewc=calTcenter(Toldc,T1step)
        rR=np.linalg.norm(Toldc-Tnewc)        
        convergence=(rR>1e-7)
    
        print "niter",niter
        print "residu",rR
        print "Temp centre",T1step[0,:].mean()
        print "Temp 1 couronne",T1step[1,:].mean()
        print "Temp 2 couronne",T1step[2,:].mean()    
        niter=niter+1
        T[0,:]=Tnewc        

# Boucle de convergence selon les angles
    niter=0
    convergence=True   
    while convergence :
        
        heat_source=fake_source(Cable,p=2,Qmax=0.)
        rhs1=mul3cols(beta_m,1-beta,beta_p,T)
        Toldc=T[0,:]
        rhs1[0,:]=Toldc
        rhs2=heat_source*dt/2/Material.Ther.density/Material.Ther.heat_capacity
        rhs2[-1,:]=rhs2[-1,:]+eta_nj*Cable.CL.Tinf
        rhs2[0,:]=0
        rhs=rhs1+rhs2
        T1step=np.empty_like(T)
        for j in range(m+1) :
            T1step[:,j]=Solution_Thomas(alpha_m[1:,j],1+alpha[:,j],alpha_p[:-1,j],rhs[:,j])
    
        Tnewc=calTcenter(Toldc,T1step)
        
        rR=np.linalg.norm(Toldc-Tnewc)        
        convergence=(rR>1e-7)
    
        T[0,:]=Tnewc
        print "niter",niter
        print "residu",rR
        print "Temp centre",T1step[0,:].mean()
        print "Temp 1 couronne",T1step[1,:].mean()
        print "Temp 2 couronne",T1step[2,:].mean()    
        
        niter=niter+1    
    
    Cable.plotMail(3,'k')
    Cable.plotData(3,data = heat_source,typeplot = 'Scatter')
    plt.title('Heat Source')

    Cable.plotMail(4,'k')
    Cable.plotData(4,data = Tinit_vec,typeplot = 'Scatter')
    plt.title('Initial Temperature')

    Cable.plotMail(5,'k')
    Cable.plotData(5,data = T1step,typeplot = 'Scatter')
    plt.title('Temperature 1st Step')
#
#    T1_mean=np.mean(T[1,:])
#    T2_mean=np.mean(T[2,:])
#    print K2C(T1step.mean())
#    
#    
## _______________________________
##/    Analytical Calculations    \_______________________________
##     
#    ro = Cable.Geom.radius              # radius of sphere (a.k.a outer radius), m
#    rs = ro/ro                          # dimensionless surface radius, (-)
#    rc = 1e-16/ro                       # dimensionless center radius, (-)
#    
#    z = np.arange(0, 3000, 0.0001)         # range to evaluate the zeta, Bi equation
#    
#    Fo_g = k[0,0]/Material.Ther.density/Material.Ther.heat_capacity*time/10/Cable.Geom.radius**2          # Fourier number, (-)
#    Bi_g = h*Cable.Geom.radius/k[0,0]          # Biot number, (-)
#
#    b = 1   # shape factor where 2 sphere, 1 cylinder, 0 slab
#    
#    # surface temperature where ro for outer surface
#    thetaRo = theta(rs, b, z, Bi_g, Fo_g)   # dimensionless temperature profile
#    To_cyl = Cable.CL.Tinf + thetaRo*(Tinit-Cable.CL.Tinf)   # convert theta to temperature in Kelvin, K
#    
#    # center temperature where r for center
#    thetaR = theta(rc, b, z, Bi_g, Fo_g)    # dimensionless temperature profile
#    Tr_cyl = Cable.CL.Tinf + thetaR*(Tinit-Cable.CL.Tinf)    # convert theta to temperature in Kelvin, K
#
#    plt.figure()
#    plt.plot(time/10,Tr_cyl,label='Surface')
#    plt.plot(time/10,To_cyl,label='Center')
#    plt.grid()
#    plt.xlabel(r"$\textrm{Time}\ \left[s\right]$")
#    plt.ylabel(r"$\textrm{Temperature}\ \left[K\right]$")
#    plt.legend()
