#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Created on Fri Jul 18 09:34:52 2014

 Author : Philippe Baucour philippe.baucour@univ-fcomte.fr
"""
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,MA 02110-1301,USA.

import matplotlib.pyplot as plt

import numpy as np
#from matplotlib import rcParams&


if __name__ == '__main__':
    #~ rcParams['text.latex.unicode']=True
    # le maillag régulier
    n=30     
    g=np.linspace(0,1,n)
    # coeff pour raffiner le maillage    
    p=10
    g0=0.5 # position réelle où l'on veut affiner
    x0=1./2./p*np.log((1+(np.exp(p)-1)*g0)/(1+(np.exp(-p)-1)*g0))
    A=np.sinh(p*x0)
    # x le maillage réel
    x=g0/A*(np.sinh((g-x0)*p)+A)
    plt.plot(g,x,'bo-')
    for xi,gi in zip(x,g) :
        plt.plot([0,1],[xi,xi],'k-')
        plt.plot([gi,gi],[0,1],'k-')
    #~ plt.grid()
    plt.ylabel(u'maillage réel $x$')
    plt.xlabel(u'maillage régulier $\gamma$')
