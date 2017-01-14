#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:14:28 2017

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt

def gen_signal(time,values,size,percent) :
    """ generate a noisy time serie define by time and values but only a percentage
    stated as percent is returned """
    tf = time[-1]
    t = np.linspace(0,tf,size)
    val=np.abs(np.interp(t,time,values)+np.random.normal(scale=1,size=size))
    val[0]=0
    sizetoremove = int(percent * size)
    Indx=np.arange(size)
    np.random.shuffle(Indx)
    t = np.delete(t,Indx[:sizetoremove])
    val = np.delete(val,Indx[:sizetoremove])
    return  t,val


if __name__  ==  '__main__' :
    
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    
    plt.close('all')
    # t in seconds
    t_s = np.array([0, 30, 100, 400, 1000, 1020, 1800]) 
    # speed in km/h * 10/36 to get m/s
    speed_s = 10/36*np.array([0, 10, 10, 140, 140, 100, 100])
    # on génère le signal en bruitant les données et en supprimant
    # quelques points (80% de perte sur 1000 = 200 pts)
    time,speed = gen_signal(t_s, speed_s, 1000, 0.8)
    plt.plot(time,speed,'gd-',label='profil de vitesse')
    plt.xlabel('Temps [s]')
    plt.ylabel('Vitesse [m/s]')
    plt.grid()    
    plt.legend()

    # calcul de la distance cumumlée on considère une vitesse moyenne
    # sur un pas de temps
    dist_cum = np.cumsum(np.diff(time)*0.5*(speed[1:]+speed[:-1]))
    
    # distance entre les poteaux
    dist_poteau = 600
    list_poteau = np.arange(0,dist_cum[-1],dist_poteau)
    t_poteau = np.interp(list_poteau,dist_cum,time[:-1])

    plt.figure()
    plt.plot(time[:-1],dist_cum,'rd-',label='Distance parcourue')
    plt.plot(t_poteau,list_poteau,'hg',label='poteaux')
    # lignes pour frimer !!
    plt.plot(np.vstack((np.zeros_like(t_poteau),t_poteau,t_poteau)), \
             np.vstack((list_poteau,list_poteau,np.zeros_like(t_poteau))),\
             'g--')
    plt.xlabel('Temps [s]')
    plt.ylabel('Distance [m]')
    plt.grid()    
    plt.legend()

    