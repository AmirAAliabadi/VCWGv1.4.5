import numpy
import math

"""
Calculate the turbulent diffusion coefficient
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Scott Krayenhoff
"""

class CdTurb:
    def __init__(self,nz,Ck,tke,dlk,Ri_b,Ri_b_cr,Ck_stable,Ck_unstable):
        self.nz = nz                   # Number of grid points in vertical column
        self.Ck = Ck                   # Coefficient used in the equation of diffusion coefficient (kappa)
        self.tke = tke                 # Turbulent kinetic energy [m^2 s^-2]
        self.dlk = dlk                 # Mixing length [m]
        self.Ri_b = Ri_b               # Bulk Richardson number
        self.Ri_b_cr = Ri_b_cr         # Critical bulk Richardson number in the urban area used to determine thermal stability
        self.Ck_stable = Ck_stable     # Constant used to determine turbulent diffusion coefficient under stable condition
        self.Ck_unstable = Ck_unstable # Constant used to determine turbulent diffusion coefficient under unstable condition

    def TurbCoeff(self):
        # set critical bulk Richardson number. It is a measure to distinguish between stable and unstable conditions

        # Define turbulent diffusion coefficient [m^2 s^-1]
        Km = numpy.zeros(self.nz+1)

        # Km should be zero at street level
        Km[0] = 0

        # Calculate turbulent diffusion coefficient [m^2 s^-1] (eq. 4.8, Krayenhoff, PhD thesis)
        # Km = Ck*lk*(TKE)^0.5
        for i in range(1,self.nz-1):
            # Discretize TKE and length scale (vertical resolution (dz) is kept constant)
            tke_m = (self.tke[i-1]+self.tke[i])/2
            dlk_m = (self.dlk[i-1]+self.dlk[i])/2

            if self.Ri_b > self.Ri_b_cr:
                # Calculate turbulent diffusion coefficient [m^2 s^-1]
                Km[i] = dlk_m * (math.sqrt(tke_m)) * self.Ck_stable

                # It is assumed that there is no mixing at the top of the domain. Thus, vertical gradient of Km is zero.
                Km[self.nz - 1] = Km[self.nz - 2]
                Km[self.nz] = Km[self.nz - 2]
            else:
                # Calculate turbulent diffusion coefficient [m^2 s^-1]
                Km[i] = dlk_m * (math.sqrt(tke_m)) * self.Ck_unstable

                # It is assumed that there is no mixing at the top of the domain. Thus, vertical gradient of Km is zero.
                Km[self.nz - 1] = Km[self.nz - 2]
                Km[self.nz] = Km[self.nz - 2]

        return Km