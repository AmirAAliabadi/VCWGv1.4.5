from math import log, pow, exp
import numpy

"""
Calculate dew point temperature, saturation pressure, specific humidity
Developed by Bruno Bueno
Building Technology Lab; Massachusetts Institute of Technology, Cambridge, USA
Last update: March 2012
"""

# Modified version of Psychometrics by Tea Zakula
# MIT Building Technology Lab
# Input: Tdb_in, w_in, P
# Output: Tdb, w, phi, h, Tdp, v
#
# where:
# Tdb_in = dry bulb temperature [K]
# w_in = Humidity Ratio also known as specific humidity (q) [kgv kgda^-1]
# P = Atmospheric Station Pressure [Pa]
#
# Tdb: dry bulb temperature [C]
# w: Humidity Ratio also known as specific humidity (q) [kgv kgda^-1]
# phi: relative humidity [(Pw/Pws)*100]
# Tdp: dew point temperature [C]
# h: enthalpy [J kga^-1]
# v: specific volume also equal to inverse of density [m^3 kga^-1]

def psychrometrics (Tdb_in, w_in, P):
    # Change units
    c_air = 1006.   # air heat capacity, value from ASHRAE Fundamentals [J kg^-1 K^-1]
    hlg = 2501000.  # latent heat, value from ASHRAE Fundamentals [J kg^-1]
    cw  = 1860.     # water vapor heat capacity, value from ASHRAE Fundamentals [J kg^-1 K^-1]
    P = P/1000.     # convert from [Pa] to [kPa]

    # Dry bulb temperature [C]
    Tdb = Tdb_in - 273.15
    w = w_in


    # phi (RH) calculation from Tdb and w
    Pw = (w*P)/(0.621945 + w)                             # partial pressure of water vapor [kPa]
    Pws = saturation_pressure(Tdb)                        # Get saturation pressure for given Tdb [kPa]

    phi = (Pw/Pws)*100.0

    # enthalpy calculation from Tdb and w [J kga^-1]
    h = c_air*Tdb + w*(hlg+cw*Tdb)

    # specific volume calculation from Tdb and w [m^3 kga^-1]
    v = 0.287042 * (Tdb+273.15)*(1+1.607858*w)/P

    # dew point calculation from w
    # water vapor partial pressure in [kPa]
    _pw = numpy.abs((w*P)/(0.621945 + w))
    alpha = log(_pw)

    # Dew point temperature valid for Tdp between 0 C and 93 C [C]
    Tdp = 6.54 + 14.526*alpha + pow(alpha,2)*0.7389 + pow(alpha,3)*0.09486 + pow(_pw,0.1984)*0.4569

    return  Tdb, w, phi, h, Tdp, v

def saturation_pressure(Tdb_):
    # Temperature [K], limit temperature in case EPW file is corrupted
    T_min = 273.15 - 40
    T_max = 273.15 + 60
    T = max(Tdb_ + 273.15, T_min)
    T = min(T, T_max)

    # Saturation pressure [Pa]
    _Pws = exp(-1*(5.8002206e3) / T+1.3914993 + (4.8640239e-2)*T*(-1.) + (4.1764768e-5)*pow(T,2) - (1.4452093e-8)*pow(T,3) + 6.5459673*log(T))
    # Saturation pressure [kPa]
    _Pws = _Pws/1000.
    return _Pws

def moist_air_density(P,Tdb,H):
    # Moist air density [kgv m^-3] given dry bulb temperature, humidity ratio, and pressure.
    # ASHRAE Fundamentals (2005) ch. 6 eqn. 28
    # ASHRAE Fundamentals (2009) ch. 1 eqn. 28
    moist_air_density = P/(1000*0.287042*Tdb*(1.+1.607858*H))
    return moist_air_density

def HumFromRHumTemp(RH,T,P):
    # Derive Specific Humidity [kgh20 kgn202^-1] from RH, T and Pa
    # Saturation vapour pressure from ASHRAE
    C8 = -5.8002206e3
    C9 = 1.3914993
    C10 = -4.8640239e-2
    C11 = 4.1764768e-5
    C12 = -1.4452093e-8
    C13 = 6.5459673

    # Convert temperature from Celcius [C] to Kelvin [K]
    T += 273.15

    PWS = exp(C8/T + C9 + C10*T + C11 * pow(T,2) + C12 * pow(T,3) + C13 * log(T))
    PW = RH*PWS/100.0        # Vapour pressure
    W = 0.62198*PW/(P-PW)    # Specific humidity
    return W
