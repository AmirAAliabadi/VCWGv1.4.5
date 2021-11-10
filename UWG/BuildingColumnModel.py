import numpy
import math

"""
Calculate sink and source terms associated with the presence of buildings in the 1D model for momentum, heat, and TKE  
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: February 2020
Originally developed by Alberto Martilli, Scott Krayenhoff, and Negin Nazarian
"""

# explicit and implicit terms for the building
class BuildingCol:

    # number of street direction (assumed to be one)
    nd = 1

    def __init__(self,nz,dz,dt,vol,roadfrac,lambdap,lambdaf,hmean,Ck,Cp,th_ref,vx,vy,th,Cdrag,ptg,ptr,ptw,rho,nz_u,pb,ss
                 ,g,z0_road,z0_roof,SensHt_HVAC,HVAC_street_frac,HVAC_atm_frac,windMin):
        self.nz = nz                     # Number of grid points in vertical column
        self.dz = dz                     # Grid resolution [m]
        self.dt = dt                     # Time step [s]
        self.vol = vol                   # Fraction of air to total volume in each urban unit cell
        self.roadfrac = roadfrac         # assumption: roadfrac = 1
        self.lambdap = lambdap           # Plan area fraction of buildings
        self.lambdaf = lambdaf           # Ratio of wall area facing ambient wind to plan area
        self.hmean = hmean               # Average building height [m]
        self.Ck = Ck                     # Coefficient used in the equation of turbulent diffusion coefficient (kappa)
        self.Cp = Cp                     # Heat capacity of dry air [J kg^-1 K^-1]
        self.th_ref = th_ref             # Reference potential temperature [K]
        self.vx = vx                     # x component of horizontal wind velocity vector [m s^-1]
        self.vy = vy                     # y component of horizontal wind velocity vector [m s^-1]
        self.th = th                     # Potential temperature [K]
        self.Cdrag = Cdrag               # Drag coefficient due to buildings (sectional drag coefficient)
        self.ptg = ptg                   # Road surface temperature (ptg) [K].
        self.ptr = ptr                   # Roof surface temperature (ptg) [K]
        self.ptw = ptw                   # Wall surface temperature (ptg) [K]
        self.rho = rho                   # density profile [kg m^-3]
        self.nz_u = nz_u                 # Number of grid points within the canyon
        self.pb = pb                     # Probability distribution of building height (assumed to be one
                                         # within the canyon and zero above the canyon)
        self.ss = ss                     # Probability that a building has a height equal to z (assumed to be
                                         # one at average building height and zero the other heights)
        self.g = g                       # Gravitational acceleration [m s^-2]
        self.z0_road = z0_road           # Road roughness [m]
        self.z0_roof = z0_roof           # Roof roughness [m]
        self.SensHt_HVAC = SensHt_HVAC            # Sensible waste heat from building [W m^-2]
        self.HVAC_street_frac = HVAC_street_frac  # Fraction of Sensible waste heat from building released into the atmosphere at street level
        self.HVAC_atm_frac = HVAC_atm_frac        # Fraction of sensible waste heat from building released into the atmosphere
        self.windMin = windMin           # Minimum wind in the urban area [m s^-1]

    def BuildingDrag(self):
        # Define momentum and heat fluxes from horizontal surfaces
        uhb = numpy.zeros(self.nz_u+1)    # Term in momentum equation [m^2 s^-2]
        vhb = numpy.zeros(self.nz_u+1)    # Term in momentum equation [m^2 s^-2]
        ehb = numpy.zeros(self.nz_u+1)    # Term in turbulent kinetic energy equation [m^2 s^-3]
        thb = numpy.zeros(self.nz_u+1)    # Term in energy equation [K m s^-1]
        # Define momentum and heat fluxes on vertical surfaces
        uva = numpy.zeros(self.nz_u)      # Term in momentum equation [m s^-1]
        vva = numpy.zeros(self.nz_u)      # Term in momentum equation [m s^-1]
        uvb = numpy.zeros(self.nz_u)      # Term in momentum equation
        vvb = numpy.zeros(self.nz_u)      # Term in momentum equation
        tva = numpy.zeros(self.nz_u)      # Term in energy equation [m s^-1]
        tvb = numpy.zeros(self.nz_u)      # Term in energy equation [K m s^-1]
        evb = numpy.zeros(self.nz_u)      # Term in energy equation [m^3 s^-3]

        # Define friction velocity [m s^-1](for record keeping only)
        self.ustarCol = numpy.zeros(self.nz_u+1)

        # Define explicit and implicit parts of source and sink terms due to building
        self.srex_vx_h = numpy.zeros(self.nz)  # Term in momentum equation
        self.srex_vy_h = numpy.zeros(self.nz)  # Term in momentum equation
        self.srex_tke_h = numpy.zeros(self.nz) # Term in turbulent kinetic energy equation
        self.srex_th_h = numpy.zeros(self.nz)  # Term in energy equation
        self.srim_vx_v = numpy.zeros(self.nz)  # Term in momentum equation
        self.srim_vy_v = numpy.zeros(self.nz)  # Term in momentum equation
        self.srex_tke_v = numpy.zeros(self.nz) # Term in turbulent kinetic energy equation
        self.srim_th_v = numpy.zeros(self.nz)  # Term in energy equation
        self.srex_th_v = numpy.zeros(self.nz)  # Term in energy equation

        # Define surface heat fluxes [W m^-2](for record keeping only)
        self.sfr = numpy.zeros(self.nz_u+1)
        self.sfw = numpy.zeros(self.nz)

        # Calculation of ground flux (for simulation w/o probability)

        self.FluxFlatG = self.Flux_Flat(self.z0_road,self.vx[0],self.vy[0],self.th[0],self.th[1],self.th_ref[0],self.ptg,self.windMin)
        uhb[0] = self.FluxFlatG[0]
        vhb[0] = self.FluxFlatG[1]
        ehb[0] = self.FluxFlatG[2]
        thb[0] = self.FluxFlatG[3]
        self.ustarCol[0] = self.FluxFlatG[4]

        # Term in momentum equation [m s^-2]
        self.srex_vx_h[0] = (uhb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Term in momentum equation [m s^-2]
        self.srex_vy_h[0] = (vhb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Term in turbulent kinetic energy equation [m^2 s^-3]
        self.srex_tke_h[0] = (ehb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Term in energy equation [K s^-1]
        # Option a: do not consider HVAC waste heat as source/sink term
        # self.srex_th_h[0] = (thb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Option b: consider HVAC waste heat as source/sink term at the street
        # Kinematic heat flux of the HVAC waste heat should be scaled according to building footprint area and urban area in each urban unit
        # HVAC waste heat flux building x Building footprint area = HVAC waste heat flux urban x urban area
        # HVAC waste heat flux urban = (Building footprint area / urban area) x HVAC waste heat flux building
        # HVAC waste heat flux urban = (lambdap / (1-lambdap)) x HVAC waste heat flux building
        self.srex_th_h[0] = (thb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac + \
                       self.HVAC_atm_frac*self.HVAC_street_frac*(self.SensHt_HVAC/(self.rho[0]*self.Cp)/self.dz)*self.lambdap/(1-self.lambdap)

        # Calculation of fluxes of other points
        for i in range(1,self.nz_u+1):
            # At roof level for simple and non-probabilistic canyon

            if self.ss[i] > 0:
                self.FluxFlatR = self.Flux_Flat(self.z0_roof, self.vx[i],self.vy[i], self.th[i],self.th[i+1], self.th_ref[i],self.ptr,self.windMin)
                uhb[i] = self.FluxFlatR[0]
                vhb[i] = self.FluxFlatR[1]
                ehb[i] = self.FluxFlatR[2]
                thb[i] = self.FluxFlatR[3]
                self.ustarCol[i] = self.FluxFlatR[4]
            else:
                # On walls for simple and non-probabilistic canyon
                uhb[i] = 0
                vhb[i] = 0
                ehb[i] = 0
                thb[i] = 0

            # At roof level for simple and non-probabilistic canyon
            # Term in momentum equation [m s^-2]
            self.srex_vx_h[i] += (uhb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Term in momentum equation [m s^-2]
            self.srex_vy_h[i] += (vhb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Term in turbulent kinetic energy equation [m^2 s^-3]
            self.srex_tke_h[i] += (ehb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Term in energy equation [K s^-1]
            # Option a: do not consider HVAC waste heat as source/sink term
            # self.srex_th_h[i] += (thb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Option b: consider HVAC waste heat as source/sink term at the roof
            # Kinematic heat flux of the HVAC waste heat should be scaled according to building footprint area and urban area in each urban unit
            # HVAC waste heat flux building x Building footprint area = HVAC waste heat flux urban x urban area
            # HVAC waste heat flux urban = (Building footprint area / urban area) x HVAC waste heat flux building
            # HVAC waste heat flux urban = (lambdap / (1-lambdap)) x HVAC waste heat flux building
            self.srex_th_h[i] += (thb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz) + \
                           self.HVAC_atm_frac*(1-self.HVAC_street_frac)*(self.SensHt_HVAC /(self.rho[i]*self.Cp)/self.dz)*self.ss[i]*self.lambdap/(1 - self.lambdap)

        for i in range(0,self.nz_u):
            ## Calculate wall fluxes
            self.FluxWall = self.Flux_Wall(self.vx[i],self.vy[i],self.th[i],self.Cdrag[i],self.ptw,self.rho[i],self.windMin)
            uva[i] = self.FluxWall[0]
            vva[i] = self.FluxWall[1]
            uvb[i] = self.FluxWall[2]
            vvb[i] = self.FluxWall[3]
            tva[i] = self.FluxWall[4]
            tvb[i] = self.FluxWall[5]
            evb[i] = self.FluxWall[6]

            # Within the canyon for simple and non-probabilistic canyon
            # Term in momentum equation [s^-1]
            self.srim_vx_v[i] = uva[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in momentum equation [s^-1]
            self.srim_vy_v[i] = vva[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in turbulent kinetic energy equation [m^2 s^-3]
            self.srex_tke_v[i] = evb[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in energy equation [s^-1]
            self.srim_th_v[i] = tva[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in energy equation [K s^-1]
            self.srex_th_v[i] = tvb[i]*4*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd

        for i in range(0, self.nz_u + 1):
            self.sfr[i] = -self.rho[i]*self.Cp*thb[i]

        for i in range(0,self.nz_u):
            self.sfw[i] = self.rho[i]*self.Cp*(tvb[i]+tva[i]*self.th[i])


    def Flux_Flat(self,z0,vx,vy,th0,th1,th_ref,pts,windMin):

        Utot = (vx**2+vy**2)**0.5
        zz = self.dz

        Utot = max(Utot,0.1*windMin)

        # Compute bulk Richardson number using near surface temperatures
        Ri = 2 * self.g * zz * (th1 - th0) / ((th1 + th0) * (Utot ** 2))
        # Calculation from Louis, 1979 (eq. 11 and 12)
        b = 9.4
        cm = 7.4
        ch = 5.3
        R = 0.74
        a = self.Ck/math.log(zz/z0)

        if Ri > 0:
            fm = 1/((1+0.5*b*Ri)**2)
            fh = fm
        else:
            c = b*cm*a*a*(zz/z0)**0.5
            fm = 1-b*Ri/(1+c*(-Ri)**0.5)
            c = c*ch/cm
            fh = 1-b*Ri/(1+c*(-Ri)**0.5)

        fbuw = -(a**2)*(Utot**2)*fm
        fbpt = -(a**2)*Utot*(th0-pts)*fh/R

        ustar = (-fbuw)**0.5
        tstar = -fbpt/ustar

        # x component momentum flux from horizontal surfaces [m^2 s^-2]
        uhb = -(ustar**2)*vx/Utot
        # y component momentum flux from horizontal surfaces [m^2 s^-2]
        vhb = -(ustar**2)*vy/Utot
        # Heat flux from horizontal surfaces [K m s^-1]
        thb = -ustar*tstar
        # Turbulent flux of TKE from horizontal surfaces [m^2 s^-3]
        ehb = -(self.g/th_ref)*ustar*tstar

        return uhb,vhb,ehb,thb,ustar

    def Flux_Wall(self,vx,vy,th,Cdrag,ptw,rho,windMin):

        vett = (vx**2+vy**2)**0.5
        vett = max(vett, 0.1 * windMin)
        # Implicit term of x component momentum flux from vertical surfaces [m s^-1]
        uva = -Cdrag*vett
        # Implicit term of y component momentum flux from vertical surfaces [m s^-1]
        vva = -Cdrag*vett
        # Explicit term of x component momentum flux from vertical surfaces
        uvb = 0
        # Explicit term of y component momentum flux from vertical surfaces
        vvb = 0

        # Calculation for S_theta_wall in eq. 5.5 (Krayenhoff, PhD thesis)
        # Convective heat transfer coefficient [W m^-2 K^-1]
        hc = 5.678*(1.09+0.23*(vett/0.3048))
        # Using energy balance for a control volume inside the urban unit, the convective heat transfer coefficient should be limited
        if hc > ((rho*self.Cp/self.dt)*((1-self.lambdap)*self.hmean)/(4*self.lambdaf*self.dz)):
            hc = (rho*self.Cp/self.dt)*((1-self.lambdap)*self.hmean)/(4*self.lambdaf*self.dz)
        # Term in energy equation [K m s^-1]
        tvb = (hc/(rho*self.Cp))*(ptw-th)
        tva = 0

        evb = Cdrag*(abs(vett)**3)

        return uva,vva, uvb, vvb, tva, tvb, evb
