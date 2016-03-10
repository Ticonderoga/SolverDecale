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

        elif 'Func' in vars(self) and self.Geom.typeGeom == 'Rectangular' :
            print('TODO')
    
    def plotData(self,fignum,data = [],typeplot = 'Mesh') :
        # ne fonctionne qu'avec du Polar        
        # on trace ici seulement les points et non les mailles 
        plt.figure(fignum)
        plt.axis('equal')
        plt.grid(True)                    
        
        if typeplot  ==  'Scatter' :
            plt.scatter(np.outer(self.Mail.Radius,np.cos(self.Mail.Theta)),\
                np.outer(self.Mail.Radius,np.sin(self.Mail.Theta)),c = data)
            if len(plt.gcf().axes)==1 :
                plt.colorbar()
        elif typeplot  ==  'Contour' :
            data_trigint=scint.griddata((self.Mail.grid_R.flatten(),\
                self.Mail.grid_Theta.flatten()),data.T.flatten(),\
                (self.trigint_R,self.trigint_Theta))
            plt.tricontourf(self.Mail.triang,data_trigint.flatten(),100)
            if len(plt.gcf().axes)==1 :
                plt.colorbar()
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

def mul3rows(Bm,B,Bp,T) :
    result=mul3cols(Bm.T,B.T,Bp.T,T.T)
    return result.T


def calTcenter(Toldc,T) :
    vC2=Cable.Mail.val_C2[0]
    vC3=Cable.Mail.val_C3[0]
    Tm1,Tm2=Mean_rings(T)
    Tcenter=(Toldc+heat_source[0,0]*dt/Material.Ther.density/Material.Ther.heat_capacity \
            + 2*Foij[0,0]*(vC3/dGama-2*vC2/dGama**2)*Tm1 \
            + 2*Foij[0,0]*vC2/dGama**2*Tm2) \
            / (1+2*Foij[0,0]*(vC3/dGama-vC2/dGama**2))
    return Tcenter

#def calTcenter_half(Toldc,T) :
##    Tm1=T[1,:].mean()
##    Tm2=T[2,:].mean()
#    Tm1,Tm2=T[1:3,:].mean(1)
#    Tcenter=(Toldc+heat_source[0,:]*dt/2/Material.Ther.density/Material.Ther.heat_capacity \
#            + Foij[0,:]*(Cable.Func.C3(0)/dGama-2*Cable.Func.C2(0)/dGama**2)*Tm1 \
#            + Foij[0,:]*Cable.Func.C2(0)/dGama**2*Tm2) \
#            / (1+Foij[0,:]*(Cable.Func.C3(0)/dGama-Cable.Func.C2(0)/dGama**2))
#    return Tcenter

def mat2vec(M) :
    '''A way to transform a matrix into vector'''
    return np.r_[M[0,0],M[1:,:].flatten('F')]
    
def vec2mat(V,shape) :
    '''A way to transform a vector into matrix'''
    n,m=shape
    return np.r_[V[0]*np.ones((1,m+1)),V[1:].reshape((n,m+1),order='F')]

    
if __name__  ==  '__main__' :
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    tinit_cpu=tcpu.time()
    plt.close('all')
    print(30*"=")
    print("")
    print("Maillage :")
    plt.close('all')
    Cable = Maillage('Cable.cfg')
    Cable.plotMail(1,'r')
    Cable.plotData(1,1,'Mesh')
    print(30*"=")
    print("")
    print("Matériau :")
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


    tf = 1200              # secondes
    nbdt = tf*10+1            # nb points en temps
    time,dt = np.linspace(0,tf,nbdt,retstep = True)
    interval_savet = 60
    savetime=time[time%interval_savet==0]
    nbsavet=savetime.size




    
    
#=================================================================
#    Problème Thermique
#=================================================================

    Tinit = C2K(100.)
    Tinit_vec = Tinit*np.ones((Cable.Mail.nb_r,Cable.Mail.nb_angle))
    T = np.copy(Tinit_vec)
    
    k = Material.Ther.conductivity(T)
     
    SaveT=np.empty((int(Cable.Mail.nb_r),int(Cable.Mail.nb_angle), nbsavet))
    SaveT[:,:,0]=T
    iteration_time=0
    
#    [ki,kj] = split_data(k)
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
    eta_nj = 4*Bi*Foij[-1,:]/(1-fn_1)/(3+fn_1)
    eta_ij = np.zeros_like(Tinit_vec)
    eta_ij[-1,:] = eta_nj
    alpha_m_nj = -2*Foi_mj[-1,:]*(1+fn_1)/(1-fn_1)**2/(3+fn_1)
    alpha[-1,:] = -alpha_m_nj
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
    
    
    Toldc=Tinit
    
    saveniter=[]
    
    save_rA=[]
    save_rR=[]

    max_workers=min(n,m,multiprocessing.cpu_count())
    Vm_split_Horiz=np.array_split(-beta_m[1:,1:],max_workers)
    V_split_Horiz=np.array_split(1+beta[1:,:]+eta_ij[1:,:],max_workers)
    Vp_split_Horiz=np.array_split(-beta_p[1:,:-1],max_workers)

    
    tmp=np.array([e.shape[0] for e in V_split_Horiz])
    indx_split_Horiz=list(zip(np.cumsum(tmp)-tmp+1,np.cumsum(tmp)+1))

    for t in time[1:] :
        niter=0
        heat_source=fake_source(Cable,p=2,Qmax=100000.)
        rhs1=mul3rows(-alpha_m,1-alpha,-alpha_p,T)
        rhs2=heat_source*dt/2/Material.Ther.density/Material.Ther.heat_capacity
        rhs2[-1,:]=rhs2[-1,:]+eta_nj*Cable.CL.Tinf    
        rhs=rhs1+rhs2
        rhs_split_Horiz=np.array_split(rhs[1:,:],max_workers)

        T1step=T

        # angles
        with ThreadPoolExecutor(max_workers=max_workers) as exe :

            jobs=[exe.submit(Solve_bunch_tridiag,b,a,c,f) for b,a,c,f in \
                zip(Vm_split_Horiz,V_split_Horiz,Vp_split_Horiz,rhs_split_Horiz)]

        for indices,job in zip(indx_split_Horiz,jobs) :
            T1step[indices[0]:indices[1],:]=job.result()
        
#        T1step[1:,]=Solve_bunch_tridiag(-beta_m[1:,1:],1+beta[1:,:]+eta_ij[1:,:],-beta_p[1:,:-1],rhs[1:,:])

        convergence_angle=True
        while convergence_angle :            
            Tnewc=calTcenter(Toldc,T1step)
            rA=abs(Toldc-Tnewc)
            convergence_angle=(rA>1e-6) #and niter<0         
            Toldc=Tnewc
            T1step[0,:]=Toldc
            niter=niter+1
#            save_rA.append(rA)
            
        rhs1=mul3cols(beta_m,1-beta,beta_p,T1step)              
        niter_rayon=0
        convergence_rayon=True
        while convergence_rayon :
            T2step=T1step            

            rhs1[0,:]=Toldc
            rhs2[0,:]=0
            rhs=rhs1+rhs2
            
            # rayons            
            Toldc2=Toldc
            
#            for j in range(m+1) :
#                T2step[:,j]=Solvetridiag(alpha_m[1:,j],1+alpha[:,j]+eta_ij[:,j],alpha_p[:-1,j],rhs[:,j])
#                Tnewc=calTcenter(Toldc2,T2step)
#                Toldc2=Tnewc
#                rhs1[0,:]=Toldc2
#                rhs=rhs1+rhs2
#            for j in range(m+1) :
#                T2step[:,j]=Solvetridiag(alpha_m[1:,j],1+alpha[:,j]+eta_ij[:,j],alpha_p[:-1,j],rhs[:,j])
#            
            T2step=Solve_bunch_tridiag(alpha_m[1:,:].T,1+alpha.T+eta_ij.T,alpha_p[:-1,:].T,rhs.T).T
            Tnewc=calTcenter(Toldc2,T2step)
            Toldc2=Tnewc
            rhs1[0,:]=Toldc2
            rhs=rhs1+rhs2
                

            rR=abs(Toldc-Tnewc)
            convergence_rayon=(rR>1e-6) #and niter<0
#            save_rR.append(rR)
            niter_rayon=niter_rayon+1
            Toldc=Tnewc
            T2step[0,:]=Toldc
            
        T=T2step
        Toldc=T2step[0,0]
        
        #    __________________________
        #___/ Sauvegarde des résultats \________________________________
        # On sauve toutes les interval_savet s
        
        if (t%interval_savet)==0 :
            print("time elapsed : ",t," s")
            iteration_time=iteration_time+1    
            SaveT[:,:,iteration_time]=T


        
    print("Temps total :"+"{: .3f}".format(tcpu.time()-tinit_cpu))
    

#=================================================================
#    Problème Electrique
#=================================================================
    Vinit = 0.
    Vinit_mat = Vinit*np.ones((Cable.Mail.nb_r,Cable.Mail.nb_angle))
    Vinit_vec=mat2vec(Vinit_mat)
    V = np.copy(Vinit_vec)
    sigij = 1./(Material.Elec.resistivity+Material.Elec.alpha*(T-Material.Elec.Tref))
    sigi_mj,sigi_pj,sigij_m,sigij_p=padding(sigij)
    indx_cr1=np.arange(1,(m+1)*n+1,n)
    indx_cr2=np.arange(2,(m+1)*n+1,n)
    
    data=np.ones((1,(m+1)*n+1)).repeat(5, axis=0)
    offsets=[-n,-1,0,1,n]
    Melec=scsp.dia_matrix((data, offsets), shape=((m+1)*n+1, (m+1)*n+1))


# ____________________    
#/    Calcul delta    \_______________________________

    temp = Cable.Func.C1(Cable.Mail.Gama)[:,np.newaxis]*sigij/2./dGama
    temp2_m = Cable.Func.C2(Cable.Mail.Gama)[:,np.newaxis]*sigi_mj/dGama**2
    temp2_p = Cable.Func.C2(Cable.Mail.Gama)[:,np.newaxis]*sigi_pj/dGama**2
    
    delta_m = -temp+temp2_m
    delta_p =  temp+temp2_p
    

#    eta_nj = 4*Bi*Foij[-1,:]/(1-fn_1)/(3+fn_1)
#    eta_ij = np.zeros_like(Tinit_vec)
#    eta_ij[-1,:] = eta_nj
#    alpha_m_nj = -2*Foi_mj[-1,:]*(1+fn_1)/(1-fn_1)**2/(3+fn_1)
#    alpha[-1,:] = -alpha_m_nj
#    alpha[0,:] = 0
#    alpha_m[0,:] = np.inf
#    alpha_m[-1,:] = alpha_m_nj
#    alpha_p[0,:] = 0
#    alpha_p[-1,:] = np.inf

#    Cable.plotMail(3,'k')
    Cable.plotData(3,data = heat_source,typeplot = 'Contour')
    plt.title('Heat Source')

#    Cable.plotMail(4,'k')
    Cable.plotData(4,data = Tinit_vec,typeplot = 'Contour')
    plt.title('Initial Temperature')

    Cable.plotMail(5,'k')
    Cable.plotData(5,data = T,typeplot = 'Contour')
    Cable.plotData(5,data = T,typeplot = 'Scatter')    
    plt.title('Temperature 1st Step 2')

## _______________________________
##/    Analytical Calculations    \_______________________________
##     
    Ta=Cable.CL.Tinf+(Tinit-Cable.CL.Tinf)*\
        np.exp(-h*savetime/Material.Ther.density/Material.Ther.heat_capacity/Cable.Geom.radius*2)
    plt.figure(6)
    plt.plot(savetime,Ta,'r',label='Analytic')
    plt.plot(savetime,SaveT[0,0,:],'g',label='Finite Differences')
    plt.grid(True)
    plt.legend()
    plt.title('Center temperature')   