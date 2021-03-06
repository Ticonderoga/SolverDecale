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

class Maillage(ImportData) :
    # Classe de maillage initialisée via un fichier à importer
    def __init__(self,data_file,pf = 2.,pg = 3.) :
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
            plt.scatter(mykron(Cable.Mail.Radius,np.cos(Cable.Mail.Theta)),\
                mykron(Cable.Mail.Radius,np.sin(Cable.Mail.Theta)),c = data)
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

def mykron(a,b) :
    la = len(a)
    lb = len(b)
    return np.kron(a,b).reshape(la,lb)

def split_data(k) :
    """ for a matrix k split_data(k) will return all the values in between
    example :
    [ki_p,ki_m,ki_b,kj_p,kj_m,kj_b] = split_data(k)
    """

#    ki = (k[1::,:]+k[:-1:,:])/2.
#    kj = (k[:,1::]+k[:,:-1:])/2. 
#    ki_b = ki[1::,:]+ki[:-1:,:]
#    kj_b = kj[:,1::]+kj[:,:-1:]
#    
#    return [ki[1::,:],ki[:-1:,:],ki_b,kj[:,1::],kj[:,:-1:],kj_b,]

    ki = (k[1::,:]+k[:-1:,:])/2.
    kj = (k[:,1::]+k[:,:-1:])/2. 
    return ki,kj
 
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
    gn_1=Cable.Mail.val_f[-2]

    aj = Cable.Geom.angle/2*(g[idxj+1]-g[idxj-1])
    li = Cable.Geom.radius/2*(f[idxi+1]-f[idxi-1])
    cij_p = Cable.Geom.radius/2*mykron(f[idxi]+f[idxi+1],aj)
    cij_m = Cable.Geom.radius/2*mykron(f[idxi-1]+f[idxi],aj)
    sij = Cable.Geom.radius/4*mykron(li*(f[idxi+1]+2*f[idxi]+f[idxi-1]),aj)
    sij[0,:] = sij[0,:].sum()
    
    
    dGama = Cable.Mail.Gama[1]
    dPhi = Cable.Mail.Phi[1]

    plt.figure()
    plt.axis('equal')
    Cable.plotMail(3,'k')
    Cable.plotData(3,data = sij,typeplot = 'Scatter')
    Tinit = C2K(20.)
    T = Tinit*np.ones((Cable.Mail.nb_r,Cable.Mail.nb_angle))
    k = Material.Ther.conductivity(T)
#    [ki_p,ki_m,ki_b,kj_p,kj_m,kj_b] = split_data(k)
    [ki,kj] = split_data(k)
     
    tf = 1200                 # secondes
    nbdt = tf*10+1            # nb points en temps
    time,dt = np.linspace(0,tf,nbdt,retstep = True)
    Foij = k/Material.Ther.density/Material.Ther.heat_capacity*dt/Cable.Geom.radius**2
    Foij_pm = kj/Material.Ther.density/Material.Ther.heat_capacity*dt/Cable.Geom.radius**2
#    Foij_m = kj/Material.Ther.density/Material.Ther.heat_capacity*dt/Cable.Geom.radius**2
    Foi_pmj = ki/Material.Ther.density/Material.Ther.heat_capacity*dt/Cable.Geom.radius**2
#    Foi_mj = ki/Material.Ther.density/Material.Ther.heat_capacity*dt/Cable.Geom.radius**2
    h = 20.
    Bi = h*Cable.Geom.radius/k[-1,:]

# ____________________    
#/    Calcul alpha    \_______________________________

    temp = Cable.Func.C1(Cable.Mail.Gama)[:,np.newaxis]*Foij/4./dGama
    temp2_m = Cable.Func.C2(Cable.Mail.Gama)[1::,np.newaxis]*Foi_pmj/2./dGama**2
    temp2_p = Cable.Func.C2(Cable.Mail.Gama)[:-1:,np.newaxis]*Foi_pmj/2./dGama**2
    
    alpha_m = temp[1:-1:,:] - temp2_m[:-1:,:]
    alpha_p = -temp[1:-1:,:] - temp2_p[1::,:]
    alpha = temp2_m[:-1:,] + temp2_p[1::,:]
    eta_nj = 4*Bi*Foij[-1,:]/(1-fn_1)**2/(3+fn_1)
    alpha_m_nj = (-2*Foi_pmj[-1,:]*(1+fn_1)/(1-fn_1)**2/(3+fn_1))[np.newaxis,:]
    alpha = np.r_[np.zeros((1,Cable.Mail.nb_angle)),alpha,eta_nj-alpha_m_nj]
    alpha_p = np.r_[np.zeros((1,Cable.Mail.nb_angle)),alpha_p]
    alpha_m = np.r_[alpha_m,alpha_m_nj]

            
# ___________________    
#/    Calcul beta    \_______________________________
    
    temp = (np.dot(Cable.Func.C(Cable.Mail.Gama)[:,np.newaxis], \
        Cable.Func.D1(Cable.Mail.Phi)[np.newaxis,:])*Foij/4./dPhi)
    temp2_m = (np.dot(Cable.Func.C(Cable.Mail.Gama)[:,np.newaxis], \
        Cable.Func.D2(Cable.Mail.Phi)[np.newaxis,:])/2./dPhi**2)[:,1::]*Foij_pm
    temp2_p = (np.dot(Cable.Func.C(Cable.Mail.Gama)[:,np.newaxis], \
        Cable.Func.D2(Cable.Mail.Phi)[np.newaxis,:])/2./dPhi**2)[:,:-1:]*Foij_pm

    beta_m = -temp[1::,1:-1:] + temp2_m[1::,:-1:]
    beta_p = temp[1::,1:-1] + temp2_p[1::,1::]
    beta = temp2_m[1::,:-1:] + temp2_p[1::,1::]
    beta_m_nj = 4*Foij_pm[-1,1::]/np.pi**2/(3+fn_1)/ \
            ((g[idxj+1]-g[idxj-1])*(g[idxj]-g[idxj-1]))[1:-1:]
    beta_p_nj = 4*Foij_pm[-1,:-1:]/np.pi**2/(3+fn_1)/ \
            ((g[idxj+1]-g[idxj-1])*(g[idxj+1]-g[idxj]))[1:-1:]
    beta_m[-1,:] = beta_m_nj
    beta_p[-1,:] = beta_p_nj
#TOFIX
    beta = np.r_[beta,(beta_m_nj+beta_p_nj)[np.newaxis,:]]
    
    beta_m = np.c_[ np.zeros((Cable.Mail.nb_r-1,1)), \
        beta_m, \
        Foij_pm[:,-1]*Cable.Func.C(Cable.Mail.Gama)* \
        Cable.Func.D2(Cable.Mail.Phi)[-1]/dPhi**2]
        
    beta_p = np.c_[ Foij_pm[:,0]*Cable.Func.C(Cable.Mail.Gama)* \
        Cable.Func.D2(Cable.Mail.Phi)[0]/dPhi**2, \
        beta_p, \
        np.zeros((Cable.Mail.nb_r,1))]
    
    beta = np.c_[beta_m[:,0]+beta_p[:,0],beta,beta_m[:,-1]+beta_p[:,-1]]
    beta_m[-1,-1] = Foij_pm[-1,-1]*4/np.pi**2/(1-g[m-1])**2/(3+fn_1)
    beta_p[-1,-1] = 0.
    beta[-1,-1] = Foij_pm[-1,-1]*4/np.pi**2/(1-g[m-1])**2/(3+fn_1)

    T_cal_radius = np.empty_like(T)
# ______________    
#/    Calcul    \_______________________________
    nbdt=1
    for iter_time in range(nbdt):
        rhs_radius_base = np.zeros((Cable.Mail.nb_r,Cable.Mail.nb_angle))
        rhs_radius_base[-1,:] = eta_nj*Cable.CL.Tinf
        for j in range(int(Cable.Mail.nb_angle-1)) :
            
            rhs_radius = np.c_[beta_m[:,j:j+1],1-beta[:,j:j+1],beta_p[:,j:j+1]]
            T_cal_radius[:,j] = Solution_Thomas(alpha_m[:,j],\
                np.ones_like(alpha[:,j])+alpha[:,j],\
                alpha_p[:,j],rhs_radius[:,j])
