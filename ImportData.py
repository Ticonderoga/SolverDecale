#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import numpy as np
from functools import partial
#attention il y a aussi scipy.integrate
import scipy.interpolate as scinterp 

class Section(object) :
    def __init__(self,config,sec) :
        for it in config.options(sec) :
            # chaque it est la liste des variables 
            if config.get(sec,it) == '' :  # on teste si c'est vide
                vars(self)[it] = None
            elif config.get(sec,it).count(',')>=1 : # on teste si c'est une liste de param
                vars(self)[it] =map(float,config.get(sec,it).split(','))
            elif config.get(sec,it).isalpha() : # on teste si c'est une chaine
                vars(self)[it]=config.get(sec,it)
            else : # dans les autre cas c'est un float
                try :
                    vars(self)[it] = config.getfloat(sec,it)
                except ValueError : # a-priori on a un mélange float + str
                    vars(self)[it]=config.get(sec,it)
        
        for k in vars(self).keys() : 
            if k[-5:]=='param' and vars(self).get(k)<>None :
                param=vars(self).get(k)
                if type(param)==str :
                    # si param est une chaine c'est un fichier à charger
                    # on doit avoir un fichier csv avec :
                    # le delimiter= tab ou espace 
                    # les nombres au format GB
                    # la colonne 1 = abscisses
                    # la colonne 2 = ordonnées
                    param=np.loadtxt(param)
                    # afin d'eviter de lire le fichier n fois on change 
                    # la variable param en un array issu du loadtxt
                    vars(self)[k]=param
                    
                typeDyn=vars(self).get(k[:-6]+'_type')
                # on crée la fonction qui renvoie la propriété
                # la fonction partial est indispensable
                vars(self)[k[:-6]]=partial(self._dynamicProp,param=param,typeDyn=typeDyn)

    def _dynamicProp(self,arg,**keywords) :
        param=keywords.get('param')
        typeDyn=keywords.get('typeDyn')
        All_interp=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        if typeDyn == 'poly' :
            return np.polyval(param, arg)
        elif All_interp.count(typeDyn) : # il s'agit d'interpolation 
            # i.e. kind peut prendre une des valeurs suivantes
            #  ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
            f=scinterp.interp1d(param[:,0],param[:,1],kind=typeDyn)
            return f(arg)
        else :
            return "Vous n'avez pas défini de fonction"
        # A MODIFIER ici si vous voulez des fonctions particulières

class ImportData(object) :
    # il s'agit d'une classe encapsulant les sections
    def __init__(self,data_file) :
        config = ConfigParser.ConfigParser(allow_no_value=True)
        config.optionxform = str # on respecte les Majuscules / minuscules dans le fichier de cfg
        config.read(data_file) # on lit le fichier
        for sec in config.sections() :
            # boucle sur les sections
            name_sec=sec[0:4]   # on fait des shortnames
                                # ainsi Electrical devient Elec

            # pour chaque section on définit un objet Section
            vars(self)[name_sec]=Section(config,sec) 


if __name__ == '__main__':
    print "===================="
    print "Tests basiques"
    print "CuMg :"
    CuMg=ImportData('AlliageCuMg.cfg')
    print 'CuMg.Ther.conductivity(45)= ',CuMg.Ther.conductivity(45)
    print 'CuMg.Elec.resistivity= ', CuMg.Elec.resistivity
    print 'CuMg.Mech.young_modulus=', CuMg.Mech.young_modulus
    print ""
    print "CuSn :"
    CuSn=ImportData('AlliageCuSn.cfg')
    print 'CuSn.Ther.conductivity(45)= ',CuSn.Ther.conductivity(45)
    print "===================="
    print "test avec valeurs réelles"
    Eau=ImportData('Water.cfg')
    print 'Eau.Chem.Formula=',Eau.Chem.Formula
    print 'Eau.Chem.MolarMass=',Eau.Chem.MolarMass
    print "Exemple interpolation quadratic"
    print 'Eau.Ther.Conductivity(20°C)=',Eau.Ther.Conductivity(20+273.15)
    print "Exemple interpolation cubic"
    print 'Eau.Ther.Density(20°C)=',Eau.Ther.Density(20+273.15)
    print "Exemple un polynome de régression"
    print 'Eau.Ther.HeatCapacity(20°C)=',Eau.Ther.HeatCapacity(20+273.15)
