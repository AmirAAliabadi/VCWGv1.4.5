from math import log
import numpy
from Radiation_Functions import RadiationFunctions

"""
Calculate the surface heat fluxes in the urban area
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Bruno Bueno
"""

def urbflux(UCM,BEM, forc, parameter, simTime, RSM,WindUrban, bx, by,
            beta_st, A_st, U_st, FR_st, tau_alpha_e_st, eta_he_st, V_bites, c_bites, m_dot_st_f,
            c_st_f, m_dot_he_st, theta_Z, zeta_S, T_a_roof, Adv_ene_heat_mode,
            beta_pv, A_pv, eta_pv, COP_hp_min, COP_hp_max, T_hp_min, T_hp_max,
            A_wt, eta_wt, S_wt_min, S_wt_max, WindRoof, V_pcm, l_pcm, T_melt, rural):

    UCM.Q_roof = 0.         # Sensible heat flux from building roof (convective) [W m^-2]
    UCM.roofTemp = 0.       # Average urban roof temperature [K]
    UCM.wallTemp = 0.       # Average urban wall temperature [K]

    for j in xrange(len(BEM)):
        # Building energy model
        BEM[j].building.BEMCalc(UCM, BEM[j], forc, parameter, simTime, bx, by,
            beta_st, A_st, U_st, FR_st, tau_alpha_e_st, eta_he_st, V_bites, c_bites, m_dot_st_f,
            c_st_f, m_dot_he_st, theta_Z, zeta_S, T_a_roof, Adv_ene_heat_mode,
            beta_pv, A_pv, eta_pv, COP_hp_min, COP_hp_max, T_hp_min, T_hp_max,
            A_wt, eta_wt, S_wt_min, S_wt_max, WindRoof, V_pcm, l_pcm, T_melt, rural)
        # Electricity consumption of urban area [W]
        BEM[j].ElecTotal = BEM[j].building.ElecTotal * BEM[j].fl_area

        # Update element temperatures [K]
        # Calculate temperature of the floor (mass) [K]
        BEM[j].mass.layerTemp = BEM[j].mass.Conduction(simTime.dt, BEM[j].building.fluxMass,1.,0.,BEM[j].building.fluxMass)
        # Calculate surface temperatures of the roof [K]
        BEM[j].roof.SurfFlux(forc,parameter,simTime,UCM.canHum,UCM.canTemp,max(forc.wind,UCM.canWind),1.,BEM[j].building.fluxRoof,None,None,None,None)
        # Calculate surface temperatures of the wall [K]
        BEM[j].wall.SurfFlux(forc,parameter,simTime,UCM.canHum,UCM.canTemp,UCM.canWind,1.,BEM[j].building.fluxWall,None,None,None,None)

        # Calculate average wall & roof temperatures [K]
        # Depending on how many building types we have, surface temperature are determined by summing up the
        # fraction of building stock * outdoor temperature
        UCM.wallTemp = UCM.wallTemp + BEM[j].frac*BEM[j].wall.layerTemp[0]
        UCM.roofTemp = UCM.roofTemp + BEM[j].frac*BEM[j].roof.layerTemp[0]


    # Calculate surface temperature of the road [K]
    UCM.road.SurfFlux(forc,parameter,simTime,UCM.canHum,UCM.canTemp,UCM.canWind,2.,0.,None,None,None,None)
    # Update surface temperature of the road exposed to the outdoor environment [K]
    UCM.roadTemp = UCM.road.layerTemp[0]

    # Calculate latent heat flux within the canopy [W m^-2]
    if UCM.latHeat != None:
        UCM.latHeat = UCM.latHeat + UCM.latAnthrop + UCM.treeLatHeat + UCM.road.lat*(1.-UCM.bldDensity)


    # Blending height [m] (approximately the top of the roughness sublayer)
    zrUrb = 2*UCM.bldHeight
    # Reference height [m]
    zref = RSM.z[RSM.nz-1]

    # calculate friction velocity (ustar) using wind speed profile determined from column (1-D) model.
    # "WindUrban" represent mean wind speed within the canyon obtained from column (1-D) model
    # Calculate canyon air density [kg m^-3]
    dens = forc.pres/(1000*0.287042*UCM.canTemp*(1.+1.607858*UCM.canHum))
    UCM.rhoCan = dens

    # Friction velocity [m s^-1] (equation 8, Appendix A, Bueno et al.,2014)
    # ustar evaluated at twice building height given WindUrban from the 1D column model and urban aerodynamic roughness length z0u
    UCM.ustar = parameter.vk * WindUrban / log((zrUrb - UCM.l_disp) / UCM.z0u)
    # Convective scaling velocity [m s^-1] (equation 10, Appendix A, Bueno et al.,2014)
    wstar = (parameter.g*max(UCM.sensHeat,0.0)*zref/dens/parameter.cp/UCM.canTemp)**(1/3.)
    # Modified friction velocity [m s^-3]
    # For thermal convection dominated condition wstar may be more significant than ustar, take the larger of the two
    UCM.ustarMod = max(UCM.ustar,wstar)

    # Calculate exchange velocity [m s^-1]
    UCM.uExch = parameter.exCoeff*UCM.ustarMod

    return UCM,BEM, dens
