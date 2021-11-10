import os
import numpy
import math
from pprint import pprint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Rural model: The Monin-Obukhov Similarity Theory (MOST) is used to solve for the vertical profile of potential temperature
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: February 2020
'''

class RSMDef(object):

    def __init__(self, lat,lon,GMT, height, forc, parameter,z,nz,dz,WindMin_MOST,T_init,P_init,L_Pos_min,L_Pos_max,L_Neg_max,L_Neg_min,
                 ZL_Pos_cutoff,ZL_Neg_cutoff,u_star_min_MOST,z0overh_MOST,zToverz0_MOST,dispoverh_MOST,h_wind,h_temp):

        self.lat = lat                     # Latitude
        self.lon = lon                     # Longitude
        self.GMT = GMT                     # GMT
        self.z0 = z0overh_MOST*height      # Rural Aerodynamic Roughness Length [m]
        self.z_T = zToverz0_MOST*self.z0   # Rural thermodynamic length scale [m]
        self.disp = dispoverh_MOST * height# Rural displacement length [m]
        self.parameter = parameter         # Constant parameters
        self.forc = forc                   # Forcing parameters
        self.z = z                         # Grid points [m]
        self.nz = nz                       # Number of grid points in vertical column
        self.dz = dz                       # Grid resolution [m]
        self.zr_wind = h_wind              # The height at which wind speed is measured in the rural area [m]
        self.zr_temp = h_temp              # The height at which temperature is measured in the rural area [m]
        self.ur_min_MOST = WindMin_MOST    # min wind speed [m s^-1]
        self.L_Pos_min = L_Pos_min         # Minimum positive Obukhov length [m] used by the rural model
        self.L_Pos_max = L_Pos_max         # Maximum positive Obukhov length [m] used by the rural model
        self.L_Neg_max = L_Neg_max         # Maximum negative Obukhov length [m] used by the rural model
        self.L_Neg_min = L_Neg_min         # Minimum negative Obukhov length [m] used by the rural model
        self.ZL_Pos_cutoff = ZL_Pos_cutoff # Cutoff z/L for neutral to stable condition used by the rural model
        self.ZL_Neg_cutoff = ZL_Neg_cutoff # Cufoff z/L for neutral to unstable condition used by the rural model
        self.u_star_min_MOST = u_star_min_MOST       # Minimum friction velocity used by the rural model typically in the order of 0.1 * WindMin_MOST [m s^-1]

        # Initialize potential temperature profile in the rural area [K]
        self.T_rural = [T_init for x in range(self.nz)]
        # Initialize specific humidity profile in the rural area [kg kg^-1]
        self.q_rural = [0 for x in range(self.nz)]
        # Initialize pressure profile in the rural area [Pa]
        self.presProf = [P_init for x in range(self.nz)]
        # Initialize real temperature profile in the rural area [K]
        self.tempRealProf = [T_init for x in range(self.nz)]
        # Initialize density profile at the center of layers in the rural area [kg m^-3]
        self.densityProfC = [None for x in range(self.nz)]
        # Initialize wind speed profile in the rural area [m s^-1]
        self.windProf = [1 for x in range(self.nz)]

    def MOST(self,rural):

        # Calculate forcing specific humidity
        C8 = -5.8002206e3
        C9 = 1.3914993
        C10 = -4.8640239e-2
        C11 = 4.1764768e-5
        C12 = -1.4452093e-8
        C13 = 6.5459673
        # Define set pressures and altitudes from https://www.weather.gov/epz/wxcalc_pressurealtitude
        pdata = [950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000, 1005, 1010, 1015, 1020]
        hdata = [540.1, 496.4, 452.8, 409.4, 366.3, 323.2, 280.4, 237.8, 195.3, 153, 110.8, 68.9, 27.1, -14.6, -56]
        pApprox = interp1d(hdata, pdata)
        P = 100 * pApprox(1.5)
        T0 = self.forc.temp
        # saturation vapor pressure similar to Clausius-Clapayron equation
        PWS0 = numpy.exp(C8 / T0 + C9 + C10 * T0 + C11 * pow(T0, 2) + C12 * pow(T0, 3) + C13 * numpy.log(T0))

        PW0 = self.forc.hum * PWS0 / 100.0  # Vapour pressure
        W0 = 0.62198 * PW0 / (P - PW0)
        Forc_q = W0 / (1 + W0)  # 4. Specific humidity


        # Calculate pressure profile [Pa]
        for iz in reversed(range(self.nz)[1:]):
            self.presProf[iz - 1] = (math.pow(self.presProf[iz], self.parameter.r / self.parameter.cp) + \
                                     self.parameter.g / self.parameter.cp * (math.pow(self.forc.pres, self.parameter.r / self.parameter.cp)) * \
                                     (1. / self.T_rural[iz] + 1. / self.T_rural[iz - 1]) * \
                                     0.5 * self.dz) ** (1. / (self.parameter.r / self.parameter.cp))

        # Calculate the real temperature profile [K]
        for iz in xrange(self.nz):
            self.tempRealProf[iz] = self.T_rural[iz] * \
                                    (self.presProf[iz] / self.forc.pres) ** (self.parameter.r / self.parameter.cp)

        # Calculate the density profile [kg m^-3]
        for iz in xrange(self.nz):
            self.densityProfC[iz] = self.presProf[iz] / self.parameter.r / self.tempRealProf[iz]

        # Temperature at the lower bound of integral
        # Option 1: extrapolation
        Slope_T = (self.forc.temp-rural.T_ext)/self.z_T
        theta_lb = rural.T_ext + Slope_T*self.z_T
        # Option 2: forcing temperature
        # theta_lb = self.forc.temp

        # Number of iteration used to determine friction velocity
        N_iter = 5

        self.ur_wind = self.forc.wind
        # Check minimum wind speed
        if self.ur_wind < self.ur_min_MOST:
            self.ur_wind = self.ur_min_MOST

        g = 9.81
        # Density at the reference level
        rho_0 = self.forc.pres / (287 * theta_lb)


        # Calculate turbulent heat flux [K m s^-1]
        self.wt = rural.sens / (rho_0 * self.parameter.cp)

        # Calculate latent heat flux []
        # Using latent heat flux calculated in UWG
        self.wq = rural.lat / (rho_0 * self.parameter.lv)

        # Option 1: Calculate friction velocity iteratively considering roughness and stability effect
        # Friction velocity at the reference level
        u_star_init = self.parameter.vk * self.ur_wind / math.log((self.zr_wind-self.disp) / self.z0)
        # Obukhov length at the reference level
        L_init = -(theta_lb * u_star_init ** 3) / (self.parameter.vk * g * self.wt)

        #Iterate until we converge at a stability-corrected friction velocity
        #Iterate maximum number of times
        for j in range(1, N_iter):
            #Stable
            # Solve equation using Businger et al. 1971 and Dyer 1970
            if (self.zr_wind-self.disp) / L_init > self.ZL_Pos_cutoff:
                # Option 1: u_star based on roughness length and stability
                # self.u_star = (self.ur_wind * self.parameter.vk)/(math.log((self.zr_wind-self.disp) / self.z0)+(4.7*(self.zr_wind-self.disp))/L_init-4.7*self.z0/L_init)
                # Option 2: fraction of mean wind
                self.u_star = 0.07*self.ur_wind


            #Unstable
            elif (self.zr_wind-self.disp) / L_init < self.ZL_Neg_cutoff:

                # Solve equation using parameterization in Paulson 1970
                alfa2 = (1-16*(self.zr_wind-self.disp)/L_init)**(0.25)
                alfa1 = (1 - 16 * (self.z0) / L_init) ** (0.25)
                ksi2 = -2*math.log((1+alfa2)/2)-math.log((1+alfa2**2)/2)+2*math.atan(alfa2)-math.pi/2
                ksi1 = -2*math.log((1+alfa1)/2)-math.log((1+alfa1**2)/2)+2*math.atan(alfa1)-math.pi/2
                self.u_star = self.ur_wind*self.parameter.vk/(math.log((self.zr_wind-self.disp)/self.z0)+ksi2-ksi1)

            #Neutral
            else:
                self.u_star = self.parameter.vk * self.ur_wind / math.log((self.zr_wind-self.disp) / self.z0)

            if self.u_star < self.u_star_min_MOST:
                self.u_star = self.u_star_min_MOST
            # u_star_init = self.u_star
            self.L = -(theta_lb * self.u_star ** 3) / (self.parameter.vk * g * self.wt)
            L_init = self.L

        # Option 2: Consider u_star as a fraction of mean wind speed
        #self.u_star = self.parameter.vk * self.ur_wind / math.log((self.zr_wind-self.disp) / self.z0)
        #self.L = -(theta_lb * self.u_star ** 3) / (self.parameter.vk * g * self.wt)

        if self.L > 0 and self.L < self.L_Pos_min:
            self.L = self.L_Pos_min
        if self.L > 0 and self.L > self.L_Pos_max:
                self.L = self.L_Pos_max
        if self.L < 0 and self.L > self.L_Neg_max:
            self.L = self.L_Neg_max
        if self.L < 0 and self.L < self.L_Neg_min:
            self.L = self.L_Neg_min

        # Surface layer temperature scale
        self.theta_sl = -self.wt / self.u_star
        # Surface layer humidity scale
        self.q_sl = -self.wq / self.u_star

        for j in range(1, self.nz):
            # Stable
            if (self.z[j]-self.disp)  / self.L > self.ZL_Pos_cutoff:
                # Businger et al. 1971 and Dyer 1970
                self.T_rural[j] = (self.theta_sl / self.parameter.vk) * (numpy.log((self.z[j]-self.disp) / self.z_T) +
                                                                         5*((self.z[j]-self.disp) / self.L)- 5*(self.z_T/self.L)) + theta_lb

                # Specific humidity equation
                # Calculate forcing specific humidity
                # Closed formula
                self.q_rural[j] = (self.q_sl / self.parameter.vk) * (numpy.log((self.z[j]-self.disp)/self.z_T) +
                                                                     5*((self.z[j]-self.disp)/self.L) - 5*(self.z_T / self.L)) + self.forc.hum

            # Unstable
            elif (self.z[j]-self.disp)  / self.L < self.ZL_Neg_cutoff:

                # Calculate alpha for heat according to Paulson 1970 / Garratt
                alphaHMO2 = (1 - 16 * (self.z[j]-self.disp) / self.L) ** 0.5
                alphaHMO1 = (1 - 16 * self.z_T / self.L) ** 0.5
                # Calculate Psi for heat according to Paulson 1970 / Garratt
                PsiHMO2 = 2 * numpy.log((1 + alphaHMO2) / 2)
                PsiHMO1 = 2 * numpy.log((1 + alphaHMO1) / 2)
                # Calculate wind profile according to Paulson 1970
                self.T_rural[j] = (self.theta_sl / self.parameter.vk) * (numpy.log((self.z[j]-self.disp)/self.z_T) - PsiHMO2 + PsiHMO1) + theta_lb

                # Calculate alpha for heat according to Paulson 1970 / Garratt
                alphaHMO2 = (1 - 16 * (self.z[j]-self.disp) / self.L) ** 0.5
                alphaHMO1 = (1 - 16 * self.z_T / self.L) ** 0.5
                # Calculate Psi for heat according to Paulson 1970 / Garratt
                PsiHMO2 = 2 * numpy.log((1 + alphaHMO2) / 2)
                PsiHMO1 = 2 * numpy.log((1 + alphaHMO1) / 2)
                # Calculate wind profile according to Paulson 1970
                self.q_rural[j] = (self.q_sl / self.parameter.vk) * (numpy.log((self.z[j]-self.disp) / self.z_T) - PsiHMO2 + PsiHMO1) + self.forc.hum

            # Neutral
            else:
                self.T_rural[j] = theta_lb

                self.q_rural[j] = self.forc.hum