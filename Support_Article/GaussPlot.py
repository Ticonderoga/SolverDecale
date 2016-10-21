#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 08:16:02 2016

@author: phil

Program to realize figure in article 1D

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__  ==  '__main__' :
    
    x=np.linspace(-20,20,10000)
    fs=norm(loc=0,scale=1)
    y=norm.pdf(x)
    plt.plot(x,y)
