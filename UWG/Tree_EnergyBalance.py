import numpy
import math
import matplotlib.pyplot as plt

'''
Calculate tree temperature, i.e. leaves energy balance, and tree energy fluxes
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: February 2020
'''

class Tree_EB(object):
    def TreeCal(self, th,vx, vy, SWRabsTree,R, Cp,omega, leaf_width, nz, dz,h_LAD,BowenRatio_tree):

        rcp = R / Cp
        # Set leaf dimension of trees
        leaf_dim = 0.72 * leaf_width
        # Air pressure [Pa]
        pr = 101300
        # Molar heat capacity [J mol^-1 K^-1](Campbell and Norman, 1998)
        cp_mol = 29.3

        Q_HV = []
        Q_LV = []
        Tveg = numpy.zeros(nz)
        for jTree in range(0, nz):
            if dz * jTree < max(h_LAD):
                # source/sink terms of specific humidity
                wind = numpy.sqrt(vx[jTree] ** 2 + vy[jTree] ** 2)

                # Conductance for heat [mol m^-2 s^-1]
                gHa_multiplier = 0.2
                gHa = gHa_multiplier*1.4 * 0.135 * numpy.sqrt(wind / leaf_dim)
                # Convert potential air temperature to real temperature [K]
                # potential temperature = real temperature * (P0/P)^(R/cp)
                tair = th[jTree] / (pr / 1.e+5) ** (-rcp)

                # Calculate temperature of vegetation [K]
                # Using energy balance at leaves we assume net shortwave flux is balanced with sensible and latent heat fluxes
                # A more complex energy balance model can be used considering longwave radiation as well.
                Tveg[jTree] = tair - (2*omega*SWRabsTree) / (2*cp_mol*gHa*(1+(1/BowenRatio_tree)))

                # Calculate sensible and latent heat fluxes [W m^-2]
                Q_HV.append(2* cp_mol * gHa * (Tveg[jTree] - tair))
                Q_LV.append(Q_HV[jTree]/BowenRatio_tree)

        return Tveg, Q_HV, Q_LV