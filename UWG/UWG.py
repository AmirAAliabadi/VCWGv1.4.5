"""
=========================================================================
 THE VERTICAL CITY WEATHER GENERATOR (VCWG)
=========================================================================

Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: July 2020
Originally developed and edited by Bruno Bueno, A. Nakano, Lingfu Zhang, Joseph Yang, Saeran Vasanthakumar,
Leslie Norford, Julia Hidalgo, Gregoire Pigeon.
=========================================================================
"""

import os
import math
import cPickle
import copy
import Utilities
import logging
import ProgressBar
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate

from Simparam import SimParam
from Weather import  Weather
from BuildingEnergy import Building
from Material import Material
from Element import Element
from BEMDef import BEMDef
from schdef import SchDef
from Param import Param
from UCMDef import UCMDef
from Forcing import Forcing
from RSM import RSMDef
import UrbFlux
from Psychrometrics import psychrometrics
from ReadDOE import readDOE
from UrbFlux import urbflux
from ColModel import ColModel
from DragLength import Drag_Length
from DragTurb import CdTurb
from Radiation_Functions import RadiationFunctions
from Tree_EnergyBalance import Tree_EB

# For debugging only
#from pprint import pprint
#from decimal import Decimal
#pp = pprint
#dd = Decimal.from_float
class UWG(object):
    """Morph a rural EPW file to urban conditions using a file with a list of urban parameters.

    args:
        epwDir: The directory in which the rural EPW file sits.
        epwFileName: The name of the rural epw file that will be morphed.
        uwgParamDir: The directory in which the UWG Parameter File (.uwg) sits.
        uwgParamFileName: The name of the UWG Parameter File (.uwg).
        destinationDir: Optional destination directory for the morphed EPW file.
            If left blank, the morphed file will be written into the same directory
            as the rural EPW file (the epwDir).
        destinationFileName: Optional destination file name for the morphed EPW file.
            If left blank, the morphed file will append "_UWG" to the original file name.
    returns:
        newClimateFile: the path to a new EPW file that has been morphed to account
            for urban conditions.
    """

    """ Section 1 - Definitions for constants / other parameters """
    MINTHICKNESS = 0.01    # Minimum layer thickness (to prevent crashing) (m)
    MAXTHICKNESS = 0.05    # Maximum layer thickness (m)
    SOILTCOND = 1
    SOILVOLHEAT = 2e6
    SOIL = Material(SOILTCOND, SOILVOLHEAT, name="soil")  # Soil material used for soil-depth padding

    # Physical constants
    G = 9.81               # gravity [m s^-2]
    CP = 1004.             # heat capacity for air [J kg^-1 K^-1]
    VK = 0.40              # von karman constant
    R = 287.               # Gas constant dry air [J kg^-1 K^-1]
    RV = 461.5             # gas constant water vapor [J kg^-1 K^-1]
    LV = 2.26e6            # latent heat of evaporation [J kg^-1]
    SIGMA = 5.67e-08       # Stefan Boltzmann constant [W m^-2 K^-4]
    WATERDENS = 1000.      # water density [kg m^-3]
    LVTT = 2.5008e6        # Latent heat of vaporization [J kg^-1]
    TT = 273.16            # zero temperature [K]
    ESTT = 611.14
    CL = 4.218e3
    CPV = 1846.1
    B = 9.4                # Coefficients derived by Louis (1979)
    CM = 7.4
    COLBURN = math.pow((0.713/0.621), (2/3.)) # (Pr/Sc)^(2/3) for Colburn analogy in water evaporation

    # Site-specific parameters
    WGMAX = 0.005 # maximum film water depth on horizontal surfaces (m)

    # File path parameter
    RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "resources"))
    CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, epwFileName, uwgParamFileName=None, epwDir=None, uwgParamDir=None, destinationDir=None, destinationFileName=None):
         # Logger will be disabled by default unless explicitly called in tests
        self.logger = logging.getLogger(__name__)

        # User defined
        self.epwFileName = epwFileName if epwFileName.lower().endswith('.epw') else epwFileName + '.epw' # Revise epw file name if not end with epw
        self.uwgParamFileName = uwgParamFileName  # If file name is entered then will UWG will set input from .uwg file

        # If user does not overload
        self.destinationFileName = destinationFileName if destinationFileName else self.epwFileName.strip('.epw') + '_UWG.epw'
        self.epwDir = epwDir if epwDir else os.path.join(self.RESOURCE_PATH, "epw")
        self.uwgParamDir = uwgParamDir if uwgParamDir else os.path.join(self.RESOURCE_PATH,"parameters")
        self.destinationDir = destinationDir if destinationDir else os.path.join(self.RESOURCE_PATH,"epw_uwg")

        # Serialized DOE reference data
        self.readDOE_file_path = os.path.join(self.CURRENT_PATH,"readDOE.pkl")

        # EPW precision
        self.epw_precision = 1


        # init UWG variables
        self._init_param_dict = None

        # Define Simulation and Weather parameters
        self.Month = None        # starting month (1-12)
        self.Day = None          # starting day (1-31)
        self.nDay = None         # number of days
        self.dtSim = None        # simulation time step (s)
        self.dtWeather = None    # seconds (s)

        # HVAC system and internal laod
        self.autosize = None     # autosize HVAC (1 or 0)
        self.sensOcc = None      # Sensible heat from occupant
        self.LatFOcc = None      # Latent heat fraction from occupant (normally 0.3)
        self.RadFOcc = None      # Radiant heat fraction from occupant (normally 0.2)
        self.RadFEquip = None    # Radiant heat fraction from equipment (normally 0.5)
        self.RadFLight = None    # Radiant heat fraction from light (normally 0.7)

        # Define Urban microclimate parameters
        self.h_ublavg = None          # average boundary layer height [m]
        self.h_ref = None             # inversion height
        self.h_temp = None            # temperature height
        self.h_wind = None            # wind height
        self.BowenRatio_rural = None  # Bowen ration in the rural area
        self.MinWind_rural = None     # minimum wind for surface heat flux calculation in the rural area
        self.VolHeat_rural = None     # Volumetric heat capacity of soil [J m^-3 K^-1]
        self.ThermalCond_rural = None # Thermal conductivity of soil [W m^-2 K^-1]
        self.z0overh_MOST = None      # Aerodynamic roughness length over obstacle height for MOST
        self.zToverz0_MOST = None     # Thermodynamic roughness length over Aerodynamic roughness length for MOST
        self.dispoverh_MOST = None    # Displacement height over obstacle height for MOST
        self.c_exch = None            # exchange coefficient
        self.windMin = None           # min wind speed [m s^-1]
        self.h_obs = None             # rural average obstacle height

        # Urban characteristics
        self.bldHeight = None         # average building height [m]
        self.h_mix = None             # mixing height [m]
        self.charLength = None        # radius defining the urban area of study [aka. characteristic length] (m)
        self.alb_road = None          # road albedo
        self.d_road = None            # road pavement thickness
        self.sensAnth = None          # non-building sensible heat [W m^-2]
        self.latAnth = None           # non-building latent heat heat [W m^-2]

        # Fraction of building typology stock
        self.bld = None               # 16x3 matrix of fraction of building type by era

        # climate Zone
        self.zone = None

        # Vegetation parameters
        self.vegCover = None          # urban area veg coverage ratio
        self.vegStart = None          # vegetation start month
        self.vegEnd = None            # vegetation end month
        self.albVeg = None            # Vegetation albedo
        self.latGrss = None           # latent fraction of grass
        self.latTree = None           # latent fraction of tree
        self.rurVegCover = None       # rural vegetation cover

        # Define Traffic schedule
        self.SchTraffic = None

        # Define Road (Assume 0.5m of asphalt)
        self.kRoad = None             # road pavement conductivity [W m^-1 K^-1]
        self.cRoad = None             # road volumetric heat capacity [J m^-3 K^-1]

        # Define optional Building characteristics
        self.albRoof = None           # roof albedo (0 - 1)
        self.vegRoof = None           # Fraction of the roofs covered in grass/shrubs (0-1)
        self.glzR = None              # Glazing Ratio
        self.hvac = None              # HVAC TYPE; 0 = Fully Conditioned (21C-24C); 1 = Mixed Mode Natural Ventilation (19C-29C + windows open >22C); 2 = Unconditioned (windows open >22C)

        # 1-D model parameters
        self.nz = None                 # number of points
        self.nz_u = None               # number of canopy levels in the vertical
        self.dz = None                 # vertical resolution
        self.wx = None                 # distance between buildings at street level in the x direction [m]
        self.wy = None                 # distance between buildings at street level in the y direction [m]
        self.Cbw = None                # fraction of building dimension and buildings distance (bx/wx or by/wy)
        self.theta_can = None          # Canyon orientation [deg]
        self.prandtl = None            # Turbulent Prandtl number
        self.schmidt = None            # Turbulent Schmidt number
        self.HVAC_atm_frac = None      # Fraction of sensible waste heat from building released into the atmosphere
        self.HVAC_street_frac = None   # Fraction of Sensible waste heat from building released into the atmosphere at street level
        self.LAD = None                # leaf area density profile [m^2 m^-3]
        self.h_tk = None               # Height of trunk [m]
        self.Ncloud = None             # Fraction of sky covered by cloud
        self.LAI = None                # Leaf area index (LAI) [m^2 m^-2]
        self.L_Pos_min = None          # Minimum positive Obukhov length [m]
        self.L_Pos_max = None          # Maximum positive Obukhov length [m] used by the rural model
        self.L_Neg_max = None          # Maximum negative Obukhov length [m]
        self.L_Neg_min = None          # Minimum negative Obukhov length [m] used by the rural model
        self.ZL_Pos_cutoff = None      # Cutoff z/L for neutral to stable condition
        self.ZL_Neg_cutoff = None      # Cufoff z/L for neutral to unstable condition
        self.u_star_min_MOST = None    # Minimum friction velocity used by the rural model typically in the order of 0.1*WindMin [m s^-1]
        self.WindMin_MOST = None       # Minimum wind for MOST
        self.Ri_b_cr = None            # Critical bulk Richardson number in the urban area used to determine thermal stability
        self.Ck_stable = None          # Constant used to determine turbulent diffusion coefficient under stable condition
        self.Ck_unstable = None        # Constant used to determine turbulent diffusion coefficient under unstable condition
        self.nightStartHour = None     # Starting hour for the night
        self.nightEndHour = None       # Ending hour for the night
        self.z0_road = None            # Road roughness
        self.z0_roof = None            # Roof roughness
        self.cdmin = None              # Minimum diffusion coefficient
        self.C_dpdx = None             # Pressure gradient coefficient
        self.leaf_width = None         # Leaf width
        self.omega = None              # Total neighbourhood foliage clumping [non dimensional]
        self.omega_drag = None         # Effect of the foliage on the building drag coefficient
        self.eps_veg = None            # Emissivity of leaves surface
        self.albv_u = None             # Foliage scattering coefficient
        self.eps_wall = None           # Wall emissivity
        self.eps_road = None           # Road emissivity
        self.eps_roof = None           # Roof emissivity
        self.eps_rural = None          # Rural emissivity
        self.eps_bare = None           # Bare ground emissivity
        self.alb_wall = None           # Wall albedo
        self.alb_road = None           # Road albedo
        self.alb_roof = None           # Roof albedo
        self.alb_veg = None            # Vegetation (trees) albedo
        self.alb_bare = None           # Bare ground albedo
        self.soilCover = None          # Fraction of natural ground in the urban area which is covered by bare soil
        self.HighVegCover = None       # High-vegetation cover fraction
        self.alb_rural = None          # Rural albedo
        self.trees = None              # 1 for trees and 0 for no trees
        self.ftree = None              # 1 for trees and 0 for no trees
        self.radius_tree = None        # Radius of tree crown
        self.BowenRatio_tree = None    # Bowen ratio for trees
        self.distance_tree = None      # Distance between trees
        # View factor
        # nT = no trees, T = trees, g = ground, s = sky, w = wall, t = trees
        self.F_gs_nT = None
        self.F_gw_nT = None
        self.F_ww_nT = None
        self.F_wg_nT = None
        self.F_ws_nT = None
        self.F_sg_nT = None
        self.F_sw_nT = None
        self.F_gs_T = None
        self.F_gt_T = None
        self.F_gw_T = None
        self.F_ww_T = None
        self.F_wt_T = None
        self.F_wg_T = None
        self.F_ws_T = None
        self.F_sg_T = None
        self.F_sw_T = None
        self.F_st_T = None
        self.F_tg_T = None
        self.F_tw_T = None
        self.F_ts_T = None
        self.F_tt_T = None

        # Advanced Renewable Energy Parameters
        self.Adv_ene_heat_mode = None
        self.beta_st = None
        self.A_st = None
        self.U_st = None
        self.FR_st = None
        self.tau_alpha_e_st = None
        self.eta_he_st = None
        self.V_bites = None
        self.c_bites = None
        self.m_dot_st_f = None
        self.c_st_f = None
        self.m_dot_he_st = None
        self.beta_pv = None
        self.A_pv = None
        self.eta_pv = None
        self.COP_hp_min = None
        self.COP_hp_max = None
        self.T_hp_min = None
        self.T_hp_max = None
        self.A_wt = None
        self.eta_wt = None
        self.S_wt_min = None
        self.S_wt_max = None
        self.V_pcm = None
        self.l_pcm = None
        self.T_melt = None

    def __repr__(self):
        return "UWG: {} ".format(self.epwFileName)

    def is_near_zero(self,num,eps=1e-10):
        return abs(float(num)) < eps

    def read_epw(self):
        """Section 2 - Read EPW file
        properties:
            self.climateDataPath
            self.newPathName
            self._header    # header data
            self.epwinput   # timestep data for weather
            self.lat        # latitude
            self.lon        # longitude
            self.GMT        # GMT
            self.nSoil      # Number of soil depths
            self.Tsoil      # nSoil x 12 matrix for soil temperture (K)
            self.depth_soil # nSoil x 1 matrix for soil depth (m)
        """

        # Make dir path to epw file
        self.climateDataPath = os.path.join(self.epwDir, self.epwFileName)

        # Open epw file and feed csv data to climate_data
        try:
            climate_data = Utilities.read_csv(self.climateDataPath)
        except Exception as e:
            raise Exception("Failed to read epw file! {}".format(e.message))

        # Read header lines (1 to 8) from EPW and ensure TMY2 format.
        self._header = climate_data[0:8]

        # Read weather data from EPW for each time step in weather file. (lines 8 - end)
        self.epwinput = climate_data[8:]

        # Read Lat, Long (line 1 of EPW)
        self.lat = float(self._header[0][6])
        self.lon = float(self._header[0][7])
        self.GMT = float(self._header[0][8])

        # Read in soil temperature data (assumes this is always there)
        soilData = self._header[3]
        self.nSoil = int(soilData[1])                   # Number of ground temperature depths
        self.Tsoil = Utilities.zeros(self.nSoil,12)     # nSoil x 12 matrix for soil temperature [K]
        self.depth_soil = Utilities.zeros(self.nSoil,1) # nSoil x 1 matrix for soil depth [m]

        # Read monthly data for each layer of soil from EPW file
        for i in xrange(self.nSoil):
            self.depth_soil[i][0] = float(soilData[2 + (i*16)]) # get soil depth for each nSoil
            # Monthly data
            for j in xrange(12):
                self.Tsoil[i][j] = float(soilData[6 + (i*16) + j]) + 273.15 # 12 months of soil T for specific depth

        # Set new directory path for the moprhed EPW file
        self.newPathName = os.path.join(self.destinationDir, self.destinationFileName)

    def read_input(self):
        """Section 3 - Read Input File (.m, file)
        Note: UWG_Matlab input files are xlsm, XML, .m, file.
        properties:
            self._init_param_dict   # dictionary of simulation initialization parameters

            self.sensAnth           # non-building sensible heat (W/m^2)
            self.SchTraffic         # Traffice schedule

            self.BEM                # list of BEMDef objects extracted from readDOE
            self.Sch                # list of Schedule objects extracted from readDOE

        """

        uwg_param_file_path = os.path.join(self.uwgParamDir,self.uwgParamFileName)

        if not os.path.exists(uwg_param_file_path):
            raise Exception("Param file: '{}' does not exist.".format(uwg_param_file_path))

        # Open .uwg file and feed csv data to initializeDataFile
        try:
            uwg_param_data = Utilities.read_csv(uwg_param_file_path)
        except Exception as e:
            raise Exception("Failed to read .uwg file! {}".format(e.message))

        # The initialize.uwg is read with a dictionary so that users changing
        # line endings or line numbers doesn't make reading input incorrect
        self._init_param_dict = {}
        count = 0
        while  count < len(uwg_param_data):
            row = uwg_param_data[count]
            row = [row[i].replace(" ", "") for i in xrange(len(row))] # strip white spaces

            # Optional parameters might be empty so handle separately
            is_optional_parameter = (
                row != [] and \
                    (
                    row[0] == "albRoof" or \
                    row[0] == "vegRoof" or \
                    row[0] == "glzR" or \
                    row[0] == "hvac"
                    )
                )

            if row == [] or "#" in row[0]:
                count += 1
                continue
            elif row[0] == "SchTraffic":
                # SchTraffic: 3 x 24 matrix
                trafficrows = uwg_param_data[count+1:count+4]
                self._init_param_dict[row[0]] = map(lambda r: Utilities.str2fl(r[:24]),trafficrows)
                count += 4
            elif row[0] == "bld":
                #bld: 17 x 3 matrix
                bldrows = uwg_param_data[count+1:count+17]
                self._init_param_dict[row[0]] = map(lambda r: Utilities.str2fl(r[:3]),bldrows)
                count += 17
            elif row[0] == "LAD":
                # LAD profile
                LADrows = uwg_param_data[count+1:count+3]
                self._init_param_dict[row[0]] = map(lambda r: Utilities.str2fl(r[:(len(LADrows[0])-1)]),LADrows)
                count += 3
            elif is_optional_parameter:
                self._init_param_dict[row[0]] = float(row[1]) if row[1] != "" else None
                count += 1
            else:
                self._init_param_dict[row[0]] = float(row[1])
                count += 1

        ipd = self._init_param_dict

        # Define Simulation and Weather parameters
        if self.Month is None: self.Month = ipd['Month']
        if self.Day is None: self.Day = ipd['Day']
        if self.nDay is None: self.nDay = ipd['nDay']
        if self.dtSim is None: self.dtSim = ipd['dtSim']
        if self.dtWeather is None: self.dtWeather = ipd['dtWeather']

        # HVAC system and internal laod
        if self.autosize is None: self.autosize = ipd['autosize']
        if self.sensOcc is None: self.sensOcc = ipd['sensOcc']
        if self.LatFOcc is None: self.LatFOcc = ipd['LatFOcc']
        if self.RadFOcc is None: self.RadFOcc = ipd['RadFOcc']
        if self.RadFEquip is None: self.RadFEquip = ipd['RadFEquip']
        if self.RadFLight is None: self.RadFLight = ipd['RadFLight']

        # Define microclimate parameters
        if self.h_temp is None: self.h_temp = ipd['h_temp']
        if self.h_wind is None: self.h_wind = ipd['h_wind']
        if self.BowenRatio_rural is None: self.BowenRatio_rural = ipd['BowenRatio_rural']
        if self.MinWind_rural is None: self.MinWind_rural = ipd['MinWind_rural']
        if self.VolHeat_rural is None: self.VolHeat_rural = ipd['VolHeat_rural']
        if self.ThermalCond_rural is None: self.ThermalCond_rural = ipd['ThermalCond_rural']
        if self.c_exch is None: self.c_exch = ipd['c_exch']
        if self.windMin is None: self.windMin = ipd['windMin']
        if self.h_obs is None: self.h_obs = ipd['h_obs']

        # Urban characteristics
        if self.bldHeight is None: self.bldHeight = ipd['bldHeight']
        if self.h_mix is None: self.h_mix = ipd['h_mix']
        if self.charLength is None: self.charLength = ipd['charLength']
        if self.d_road is None: self.d_road = ipd['dRoad']
        if self.sensAnth is None: self.sensAnth = ipd['sensAnth']
        if self.latAnth is None: self.latAnth = ipd['latAnth']

        # climate Zone
        if self.zone is None: self.zone = ipd['zone']

        # Vegetation parameters
        if self.vegCover is None: self.vegCover = ipd['vegCover']
        if self.vegStart is None: self.vegStart = ipd['vegStart']
        if self.vegEnd is None: self.vegEnd = ipd['vegEnd']
        if self.latGrss is None: self.latGrss = ipd['latGrss']
        if self.latTree is None: self.latTree = ipd['latTree']
        if self.rurVegCover is None: self.rurVegCover = ipd['rurVegCover']

        # Define Traffic schedule
        if self.SchTraffic is None: self.SchTraffic = ipd['SchTraffic']

        # Define Road (Assume 0.5m of asphalt)
        if self.kRoad is None: self.kRoad = ipd['kRoad']
        if self.cRoad is None: self.cRoad = ipd['cRoad']

        # Building stock fraction
        if self.bld is None: self.bld = ipd['bld']

        # Optional parameters
        if self.glzR is None: self.glzR = ipd['glzR']
        if self.hvac is None: self.hvac = ipd['hvac']

        # 1-D model parameters
        if self.nz is None: self.nz = int(ipd['nz'])
        if self.nz_u is None: self.nz_u = int(ipd['nz_u'])
        if self.dz is None: self.dz = int(ipd['dz'])
        if self.wx is None: self.wx = ipd['wx']
        if self.wy is None: self.wy = ipd['wy']
        if self.Cbw is None: self.Cbw = ipd['Cbw']
        if self.theta_can is None: self.theta_can = ipd['theta_can']
        if self.prandtl is None: self.prandtl = ipd['prandtl']
        if self.schmidt is None: self.schmidt = ipd['schmidt']
        if self.HVAC_atm_frac is None: self.HVAC_atm_frac = ipd['HVAC_atm_frac']
        if self.HVAC_street_frac is None: self.HVAC_street_frac = ipd['HVAC_street_frac']
        if self.LAD is None: self.LAD = ipd['LAD']
        if self.h_tk is None: self.h_tk = ipd['h_tk']
        if self.Ncloud is None: self.Ncloud = ipd['Ncloud']
        if self.LAI is None: self.LAI = ipd['LAI']
        if self.L_Pos_min is None: self.L_Pos_min = ipd['L_Pos_min']
        if self.L_Pos_max is None: self.L_Pos_max = ipd['L_Pos_max']
        if self.L_Neg_max is None: self.L_Neg_max = ipd['L_Neg_max']
        if self.L_Neg_min is None: self.L_Neg_min = ipd['L_Neg_min']
        if self.ZL_Pos_cutoff is None: self.ZL_Pos_cutoff = ipd['ZL_Pos_cutoff']
        if self.ZL_Neg_cutoff is None: self.ZL_Neg_cutoff = ipd['ZL_Neg_cutoff']
        if self.u_star_min_MOST is None: self.u_star_min_MOST = ipd['u_star_min_MOST']
        if self.z0overh_MOST is None: self.z0overh_MOST = ipd['z0overh_MOST']
        if self.zToverz0_MOST is None: self.zToverz0_MOST = ipd['zToverz0_MOST']
        if self.dispoverh_MOST is None: self.dispoverh_MOST = ipd['dispoverh_MOST']
        if self.WindMin_MOST is None: self.WindMin_MOST = ipd['WindMin_MOST']
        if self.Ri_b_cr is None:  self.Ri_b_cr = ipd['Ri_b_cr']
        if self.Ck_stable is None: self.Ck_stable = ipd['Ck_stable']
        if self.Ck_unstable is None: self.Ck_unstable = ipd['Ck_unstable']
        if self.nightStartHour is None: self.nightStartHour = ipd['nightStartHour']
        if self.nightEndHour is None: self.nightEndHour = ipd['nightEndHour']
        if self.z0_road is None: self.z0_road = ipd['z0_road']
        if self.z0_roof is None: self.z0_roof = ipd['z0_roof']
        if self.cdmin is None: self.cdmin = ipd['cdmin']
        if self.C_dpdx is None: self.C_dpdx = ipd['C_dpdx']
        if self.leaf_width is None: self.leaf_width = ipd['leaf_width']
        if self.omega is None: self.omega = ipd['omega']
        if self.omega_drag is None: self.omega_drag = ipd['omega_drag']
        if self.eps_veg is None: self.eps_veg = ipd['eps_veg']
        if self.albv_u is None: self.albv_u = ipd['albv_u']
        if self.eps_wall is None: self.eps_wall = ipd['eps_wall']
        if self.eps_road is None: self.eps_road = ipd['eps_road']
        if self.eps_roof is None: self.eps_roof = ipd['eps_roof']
        if self.eps_rural is None: self.eps_rural = ipd['eps_rural']
        if self.eps_bare is None: self.eps_bare = ipd['eps_bare']
        if self.alb_wall is None: self.alb_wall = ipd['alb_wall']
        if self.alb_road is None: self.alb_road = ipd['alb_road']
        if self.alb_roof is None: self.alb_roof = ipd['alb_roof']
        if self.alb_veg is None: self.alb_veg = ipd['alb_veg']
        if self.alb_bare is None: self.alb_bare = ipd['alb_bare']
        if self.soilCover is None: self.soilCover = ipd['soilCover']
        if self.HighVegCover is None: self.HighVegCover = ipd['HighVegCover']
        if self.alb_rural is None: self.alb_rural = ipd['alb_rural']
        if self.trees is None: self.trees = ipd['trees']
        if self.ftree is None: self.ftree = ipd['ftree']
        if self.radius_tree is None: self.radius_tree = ipd['radius_tree']
        if self.BowenRatio_tree is None: self.BowenRatio_tree = ipd['BowenRatio_tree']
        if self.distance_tree is None: self.distance_tree = ipd['distance_tree']
        if self.F_gs_nT is None: self.F_gs_nT = ipd['F_gs_nT']
        if self.F_gw_nT is None: self.F_gw_nT = ipd['F_gw_nT']
        if self.F_ww_nT is None: self.F_ww_nT = ipd['F_ww_nT']
        if self.F_wg_nT is None: self.F_wg_nT = ipd['F_wg_nT']
        if self.F_ws_nT is None: self.F_ws_nT = ipd['F_ws_nT']
        if self.F_sg_nT is None: self.F_sg_nT = ipd['F_sg_nT']
        if self.F_sw_nT is None: self.F_sw_nT = ipd['F_sw_nT']
        if self.F_gs_T is None: self.F_gs_T = ipd['F_gs_T']
        if self.F_gt_T is None: self.F_gt_T = ipd['F_gt_T']
        if self.F_gw_T is None: self.F_gw_T = ipd['F_gw_T']
        if self.F_ww_T is None: self.F_ww_T = ipd['F_ww_T']
        if self.F_wt_T is None: self.F_wt_T = ipd['F_wt_T']
        if self.F_wg_T is None: self.F_wg_T = ipd['F_wg_T']
        if self.F_ws_T is None: self.F_ws_T = ipd['F_ws_T']
        if self.F_sg_T is None: self.F_sg_T = ipd['F_sg_T']
        if self.F_sw_T is None: self.F_sw_T = ipd['F_sw_T']
        if self.F_st_T is None: self.F_st_T = ipd['F_st_T']
        if self.F_tg_T is None: self.F_tg_T = ipd['F_tg_T']
        if self.F_tw_T is None: self.F_tw_T = ipd['F_tw_T']
        if self.F_ts_T is None: self.F_ts_T = ipd['F_ts_T']
        if self.F_tt_T is None: self.F_tt_T = ipd['F_tt_T']

        # Advanced Renewable Energy Parameters
        if self.Adv_ene_heat_mode is None: self.Adv_ene_heat_mode = ipd['Adv_ene_heat_mode']
        if self.beta_st is None: self.beta_st = ipd['beta_st']
        if self.A_st is None: self.A_st = ipd['A_st']
        if self.U_st is None: self.U_st = ipd['U_st']
        if self.FR_st is None: self.FR_st = ipd['FR_st']
        if self.tau_alpha_e_st is None: self.tau_alpha_e_st = ipd['tau_alpha_e_st']
        if self.eta_he_st is None: self.eta_he_st = ipd['eta_he_st']
        if self.V_bites is None: self.V_bites = ipd['V_bites']
        if self.c_bites is None: self.c_bites = ipd['c_bites']
        if self.m_dot_st_f is None: self.m_dot_st_f = ipd['m_dot_st_f']
        if self.c_st_f is None: self.c_st_f = ipd['c_st_f']
        if self.m_dot_he_st is None: self.m_dot_he_st = ipd['m_dot_he_st']
        if self.beta_pv is None: self.beta_pv = ipd['beta_pv']
        if self.A_pv is None: self.A_pv = ipd['A_pv']
        if self.eta_pv is None: self.eta_pv = ipd['eta_pv']
        if self.COP_hp_min is None: self.COP_hp_min = ipd['COP_hp_min']
        if self.COP_hp_max is None: self.COP_hp_max = ipd['COP_hp_max']
        if self.T_hp_min is None: self.T_hp_min = ipd['T_hp_min']
        if self.T_hp_max is None: self.T_hp_max = ipd['T_hp_max']
        if self.A_wt is None: self.A_wt = ipd['A_wt']
        if self.eta_wt is None: self.eta_wt = ipd['eta_wt']
        if self.S_wt_min is None: self.S_wt_min = ipd['S_wt_min']
        if self.S_wt_max is None: self.S_wt_max = ipd['S_wt_max']
        if self.V_pcm is None: self.V_pcm = ipd['V_pcm']
        if self.l_pcm is None: self.l_pcm = ipd['l_pcm']
        if self.T_melt is None: self.T_melt = ipd['T_melt']

    def set_input(self):
        """ Set inputs from .uwg input file if not already defined, the check if all
        the required input parameters are there.
        """

        # If a uwgParamFileName is set, then read inputs from .uwg file.
        # User-defined class properties will override the inputs from the .uwg file.
        if self.uwgParamFileName is not None:
            print "\nReading uwg file input."
            self.read_input()
        else:
            print "\nNo .uwg file input."

        # Required parameters
        is_defined = (type(self.Month) == float or type(self.Month) == int) and \
            (type(self.Day) == float or type(self.Day) == int) and \
            (type(self.nDay) == float or type(self.nDay) == int) and \
            type(self.dtSim) == float and type(self.dtWeather) == float and \
            (type(self.autosize) == float or type(self.autosize) == int) and \
            type(self.sensOcc) == float and type(self.LatFOcc) == float and \
            type(self.RadFOcc) == float and type(self.RadFEquip) == float and \
            type(self.RadFLight) == float and\
            type(self.h_temp) == float and type(self.h_wind) == float and \
            type(self.windMin) == float and type(self.h_obs) == float and \
            type(self.bldHeight) == float and type(self.h_mix) == float and \
            type(self.charLength) == float and \
            type(self.d_road) == float and type(self.sensAnth) == float and \
            type(self.latAnth) == float and type(self.bld) == type([]) and \
            self.is_near_zero(len(self.bld)-16.0) and \
            (type(self.zone) == float or type(self.zone) == int) and \
            (type(self.vegStart) == float or type(self.vegStart) == int) and \
            (type(self.vegEnd) == float or type(self.vegEnd) == int) and \
            type(self.latTree) == float and type(self.rurVegCover) == float and \
            type(self.kRoad) == float and type(self.cRoad) == float and \
            type(self.SchTraffic) == type([]) and self.is_near_zero(len(self.SchTraffic)-3.0)

        if not is_defined:
            raise Exception("The required parameters have not been defined correctly. Check input parameters and try again.")

        # Modify zone to be used as python index
        self.zone = int(self.zone)-1

    def instantiate_input(self):
        """Section 4 - Create UWG objects from input parameters

            self.simTime            # simulation time parameter obj
            self.weather            # weather obj for simulation time period
            self.forcIP             # Forcing obj
            self.forc               # Empty forcing obj
            self.geoParam           # geographic parameters obj
            self.RSM                # Rural site & vertical diffusion model obj
            self.USM                # Urban site & vertical diffusion model obj
            self.UCM                # Urban canopy model obj
            self.UBL                # Urban boundary layer model

            self.road               # urban road element
            self.rural              # rural road element

            self.soilindex1         # soil index for urban rsoad depth
            self.soilindex2         # soil index for rural road depth

            self.BEM                # list of BEMDef objects
            self.Sch                # list of Schedule objects
        """

        # Note bx and by mst be exactly the same; this requires wx and wy to be exactly the same
        self.bx = self.Cbw * self.wx
        self.by = self.Cbw * self.wy
        # Define variables used by UWG model; we need to interpret the 3D urban canyon into a 2D urban canyon
        # For 2D interpretation, use geometric average of x and y directions
        self.verToHor = self.bldHeight / (self.bx + self.wx)
        self.bldDensity = numpy.sqrt(self.bx * self.by) / (numpy.sqrt((self.bx + self.wx) * (self.by + self.wy)))
        # Define variables used by urban column model; we need to consider a 3D urban canyon
        # for lambdaf, use geometric average of x and y directions
        self.lambdap = (self.bx * self.by) / ((self.bx + self.wx) * (self.by + self.wy))
        self.lambdaf = numpy.sqrt((self.bldHeight * self.bx) * (self.bldHeight * self.by)) / (
                (self.bx + self.wx) * (self.by + self.wy))

        climate_file_path = os.path.join(self.epwDir, self.epwFileName)
        self.simTime = SimParam(self.dtSim,self.dtWeather,self.Month,self.Day,self.nDay)  # simulation time parameters
        self.weather = Weather(climate_file_path,self.simTime.timeInitial,self.simTime.timeFinal) # weather file data for simulation time period
        self.forcIP = Forcing(self.weather.staTemp,self.weather) # initialized Forcing class
        self.forc = Forcing() # empty forcing class
        # Initialize geographic Param and Urban Boundary Layer Objects
        maxdx = 250.;            # max dx [m]

        self.geoParam = Param(self.h_temp,self.h_wind,\
            self.latTree,self.latGrss,self.alb_veg,self.vegStart,self.vegEnd,\
            self.nightStartHour,self.nightEndHour,self.windMin,self.WGMAX,self.c_exch,maxdx,self.G,self.CP,self.VK,self.R,\
            self.RV,self.LV,math.pi,self.SIGMA,self.WATERDENS,self.LVTT,self.TT,self.ESTT,self.CL,\
            self.CPV,self.B, self.CM, self.COLBURN)

        # Defining road
        asphalt = Material(self.kRoad,self.cRoad,'asphalt')
        road_T_init = 293.
        road_horizontal = 1
        road_veg_coverage = min(self.vegCover*(1-self.bldDensity),1.) # fraction of surface vegetation coverage

        # define road layers
        road_layer_num = int(math.ceil(self.d_road/0.05))
        thickness_vector = map(lambda r: 0.05, range(road_layer_num))
        material_vector = map(lambda n: asphalt, range(road_layer_num))

        self.road = Element(thickness_vector,material_vector,road_veg_coverage,road_T_init,road_horizontal,name="urban_road")

        self.rural = copy.deepcopy(self.road)
        self.rural.vegCoverage = self.rurVegCover
        self.rural._name = "rural_road"

        # Define BEM for each DOE type (read the fraction)
        if not os.path.exists(self.readDOE_file_path):
            raise Exception("readDOE.pkl file: '{}' does not exist.".format(readDOE_file_path))

        readDOE_file = open(self.readDOE_file_path, 'rb') # open pickle file in binary form
        refDOE = cPickle.load(readDOE_file)
        refBEM = cPickle.load(readDOE_file)
        refSchedule = cPickle.load(readDOE_file)
        readDOE_file.close()

        # Define building energy models
        k = 0
        r_glaze = 0             # Glazing ratio for total building stock
        SHGC = 0                # SHGC addition for total building stock
        h_floor = 3.05          # average floor height [m]

        # total building floor area
        total_urban_bld_area = math.pow(self.charLength,2)*self.bldDensity*self.bldHeight/h_floor
        area_matrix = Utilities.zeros(16,3)

        self.BEM = []           # list of BEMDef objects
        self.Sch = []           # list of Schedule objects

        for i in xrange(16):    # 16 building types
            for j in xrange(3): # 3 built eras
                if self.bld[i][j] > 0.:
                    # Add to BEM list
                    self.BEM.append(refBEM[i][j][self.zone])
                    #print(self.BEM)
                    self.BEM[k].frac = self.bld[i][j]
                    self.BEM[k].fl_area = self.bld[i][j] * total_urban_bld_area

                    # Overwrite with optional parameters if provided
                    if self.glzR:
                        self.BEM[k].building.glazingRatio = self.glzR
                    if self.albRoof:
                        self.BEM[k].roof.albedo = self.albRoof
                    if self.vegRoof:
                        self.BEM[k].roof.vegCoverage = self.vegRoof

                    # Keep track of total urban r_glaze, SHGC, and alb_wall for UCM model
                    r_glaze = r_glaze + self.BEM[k].frac * self.BEM[k].building.glazingRatio ##
                    SHGC = SHGC + self.BEM[k].frac * self.BEM[k].building.shgc
                    #alb_wall = alb_wall + self.BEM[k].frac * self.BEM[k].wall.albedo

                    # Add to schedule list
                    self.Sch.append(refSchedule[i][j][self.zone])
                    k += 1

        # ==============================================================================================================
        # 1-D Model (Sec.1 Start): define constant parameters and Initialize variables
        # ==============================================================================================================
        # define probabilities
        # "pb(z)" Probability that a building has a height greater or equal to z (In the current version of the model a simple
        # canyon is considered. So, "pb" is one within the canyon and zero above the canyon.)
        # "ss(z)" Probability that a building has a height equal to z (In the current version of the model a simple
        # canyon is considered so this probability is one at building average height h mean (nz_u) but zero elsewhere.)
        self.pb = numpy.zeros(self.nz + 1)
        self.ss = numpy.zeros(self.nz + 1)
        self.ss[self.nz_u] = 1
        for i in range(0, self.nz + 1):
            if i <= self.nz_u:
                self.pb[i] = 1
            else:
                self.pb[i] = 0

        # Generate mesh for the column (1-D) model: grid resolution is kept constant over the domain [m]
        self.z = numpy.linspace(0, self.nz * self.dz, self.nz + 1)

        # vol: volume fraction of air in each urban unit cell
        self.vol = numpy.zeros(self.nz)
        # sf: fraction of air at the interface between cells (sf) [please verify, not sure what we are doing here!]
        self.sf = numpy.zeros(self.nz)

        for i in range(0, self.nz):
            self.vol[i] = 1-self.lambdap*self.pb[i]
            # "sf" is calculated from Nazarian's code (https://github.com/nenazarian/MLUCM/blob/master/Column_Model/column_lkPro.f90)
            self.sf[i] = 1-self.lambdap*self.ss[i]

        # Coefficient for the destruction of turbulent dissipation rate
        self.Ceps = 1 / 1.14

        # Coefficient used in the equation of diffusion coefficient
        self.Ck = 0.4

        # Coefficient which will be used to determine length scales
        # Option 1:
        # self.Cmu = 0.09
        # Option 2: Nazarian et al., 2020 (GMD)
        self.Cmu = numpy.zeros(self.nz)
        for i in range(0,self.nz):
            if self.z[i]/self.bldHeight <= 1:
                self.Cmu[i] = max(0.05,-1.6*self.lambdap**2+0.75*self.lambdap+0.022)
            elif self.z[i]/self.bldHeight > 1:
                self.Cmu[i] = 0.05

        # Calculate section drag coefficient (Cdrag) due to buildings
        # Calculate "Cdrag" based on Krayenhoff's code
        Cdrag_multiplier = 1
        DragLength = Drag_Length(self.nz, self.nz_u, self.z, self.lambdap, self.lambdaf, self.bldHeight, self.Ceps, self.Ck, self.Cmu, self.pb)
        self.dlk = DragLength.Length_Scale()[0]
        self.dls = DragLength.Length_Scale()[1]
        self.Cdrag = Cdrag_multiplier*DragLength.Drag_Coef()

        # Reference site class
        self.RSM = RSMDef(self.lat,self.lon,self.GMT,self.h_obs,self.forc,self.geoParam,self.z,self.nz,self.dz,self.WindMin_MOST,
                          self.weather.staTemp[0],self.weather.staPres[0],self.L_Pos_min,self.L_Pos_max,
                          self.L_Neg_max,self.L_Neg_min,self.ZL_Pos_cutoff,self.ZL_Neg_cutoff,self.u_star_min_MOST,self.z0overh_MOST,
                          self.zToverz0_MOST,self.dispoverh_MOST,self.h_wind,self.h_temp)

        T_init = self.weather.staTemp[0]
        H_init = self.weather.staHum[0]

        # Create Leaf area density (LAD) [m^2 m^-3] function by interpolating within the data.
        # Tree height should be equal or less than average building height.
        # Vegetation only considered in canyon column
        self.h_LAD = self.LAD[0]
        LAD = self.LAD[1]
        self.f_LAD = interp1d(self.h_LAD, LAD)

        # define variables including x and y components of wind speed, turbulent kinetic energy, potential temperature
        # specific humidity and reference temperature
        self.vx = numpy.zeros(self.nz)
        self.vy = numpy.zeros(self.nz)
        self.tke = numpy.zeros(self.nz)
        self.th = numpy.zeros(self.nz)
        self.qn = numpy.zeros(self.nz)
        self.th_ref = numpy.zeros(self.nz)
        self.Tveg = numpy.zeros(self.nz)
        # Initialize variables
        for i in range(0, self.nz):
            self.vx[i] = 0.1             # x component of horizontal wind speed [m s^-1]
            self.vy[i] = 0.1             # y component of horizontal wind speed [m s^-1]
            self.tke[i] = 0.15           # Turbulent kinetic energy [m^2 s^-2]
            self.th[i] = 300             # Potential temperature [K]
            self.th_ref[i] = 300         # Reference potential temperature [K]
            self.qn[i] = H_init          # Specific humidity [kgv kga^-1]
            self.Tveg[i] = 300           # vegetation temperature [K]

        # ==============================================================================================================
        # 1-D Model (Sec.1 End)
        # ==============================================================================================================

        self.UCM = UCMDef(self.bldHeight, self.bldDensity, self.verToHor, self.HighVegCover, self.sensAnth,self.latAnth,
                          T_init, H_init,self.weather.staUmod[0], self.geoParam, r_glaze, SHGC, self.alb_wall, self.road,
                          self.alb_road,self.vegCover,self.lambdap)
        self.UCM.h_mix = self.h_mix
        # Initial Value for Roof, Road and wall Temperatures of the canyon [K]
        self.UCM.roofTemp = 292
        self.UCM.roadTemp = 292
        self.UCM.wallTemp = 292

        # Define Road Element & buffer to match ground temperature depth
        roadMat, newthickness = procMat(self.road,self.MAXTHICKNESS,self.MINTHICKNESS)

        for i in xrange(self.nSoil):
            # if soil depth is greater than the thickness of the road
            # we add new slices of soil at max thickness until road is greater or equal

            is_soildepth_equal = self.is_near_zero(self.depth_soil[i][0] - sum(newthickness),1e-15)

            if is_soildepth_equal or (self.depth_soil[i][0] > sum(newthickness)):
                while self.depth_soil[i][0] > sum(newthickness):
                    newthickness.append(self.MAXTHICKNESS)
                    roadMat.append(self.SOIL)
                self.soilindex1 = i
                break

        self.road = Element(newthickness, roadMat,\
            self.road.vegCoverage, self.road.layerTemp[0], self.road.horizontal, self.road._name)

        # Define Rural Element
        ruralMat, newthickness = procMat(self.rural,self.MAXTHICKNESS,self.MINTHICKNESS)

        for i in xrange(self.nSoil):
            # if soil depth is greater than the thickness of the road
            # we add new slices of soil at max thickness until road is greater or equal

            is_soildepth_equal = self.is_near_zero(self.depth_soil[i][0] - sum(newthickness),1e-15)

            if is_soildepth_equal or (self.depth_soil[i][0] > sum(newthickness)):
                while self.depth_soil[i][0] > sum(newthickness):
                    newthickness.append(self.MAXTHICKNESS)
                    ruralMat.append(self.SOIL)

                self.soilindex2 = i
                break

        self.rural = Element(newthickness,\
            ruralMat,self.rural.vegCoverage,self.rural.layerTemp[0],self.rural.horizontal, self.rural._name)
        self.rural.layerVolHeat = [self.VolHeat_rural for i in range(11)]
        self.rural.layerThermalCond = [self.ThermalCond_rural for i in range(11)]
    def hvac_autosize(self):
        """ Section 6 - HVAC Autosizing (unlimited cooling & heating) """

        for i in xrange(len(self.BEM)):
            if self.is_near_zero(self.autosize)==False:
                self.BEM[i].building.coolCap = 9999.
                self.BEM[i].building.heatCap = 9999.

    def simulate(self):
        """ Section 7 - UWG main section

            self.N                  # Total hours in simulation
            self.ph                 # per hour
            self.dayType            # 3=Sun, 2=Sat, 1=Weekday
            self.ceil_time_step     # simulation time step (dt) fitted to weather file time step

            # Output of object instance vector
            self.WeatherData        # Nx1 vector of forc instance
            self.UCMData            # Nx1 vector of UCM instance
            self.UBLData            # Nx1 vector of UBL instance
            self.RSMData            # Nx1 vector of RSM instance
            self.USMData            # Nx1 vector of USM instance
        """

        self.N = int(self.simTime.days * 24)       # total number of hours in simulation
        n = 0                                      # weather time step counter
        self.ph = self.simTime.dt/3600.            # dt is simulation time step in seconds, ph is simulation time step in hours

        self.WeatherData = [None for x in xrange(self.N)]
        self.UCMData = [None for x in xrange(self.N)]
        self.RSMData = [None for x in xrange(self.N)]

        print '\nRunning VCWG for {} days from {}/{}.\n'.format(
            int(self.nDay), int(self.Month), int(self.Day))
        self.logger.info("Start simulation")

        # Start progress bar at zero
        ProgressBar.print_progress(0, 100.0, prefix = "Progress:", bar_length = 25)

        # define variables that are used to store data
        iO_new = 0
        Output_TimInd_new = [int(i * (60 / self.simTime.dt) * 60) for i in range(1, int(24 * self.simTime.days + 1))]
        U_hourly = numpy.zeros((len(Output_TimInd_new), self.nz))
        V_hourly = numpy.zeros((len(Output_TimInd_new), self.nz))
        Tu_hourly = numpy.zeros((len(Output_TimInd_new), self.nz))
        Tr_hourly = numpy.zeros((len(Output_TimInd_new), self.nz))
        q_hourly = numpy.zeros((len(Output_TimInd_new), self.nz))
        TKE_hourly = numpy.zeros((len(Output_TimInd_new), self.nz))
        sensWaste_hourly = numpy.zeros(len(Output_TimInd_new))
        dehumDemand_hourly = numpy.zeros(len(Output_TimInd_new))
        QWater_hourly = numpy.zeros(len(Output_TimInd_new))
        QGas_hourly = numpy.zeros(len(Output_TimInd_new))
        sensCoolDemand_hourly = numpy.zeros(len(Output_TimInd_new))
        coolConsump_hourly = numpy.zeros(len(Output_TimInd_new))
        sensHeatDemand_hourly = numpy.zeros(len(Output_TimInd_new))
        heatConsump_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_st_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_he_st_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_bites_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_hp_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_recovery_hourly = numpy.zeros(len(Output_TimInd_new))
        W_hp_hourly = numpy.zeros(len(Output_TimInd_new))
        W_pv_hourly = numpy.zeros(len(Output_TimInd_new))
        COP_hp_hourly = numpy.zeros(len(Output_TimInd_new))
        indoorTemp_hourly = numpy.zeros(len(Output_TimInd_new))
        T_st_f_i_hourly = numpy.zeros(len(Output_TimInd_new))
        T_st_f_o_hourly = numpy.zeros(len(Output_TimInd_new))
        T_he_st_i_hourly = numpy.zeros(len(Output_TimInd_new))
        T_he_st_o_hourly = numpy.zeros(len(Output_TimInd_new))
        T_bites_hourly = numpy.zeros(len(Output_TimInd_new))
        W_wt_hourly = numpy.zeros(len(Output_TimInd_new))
        f_pcm_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_waterSaved_hourly = numpy.zeros(len(Output_TimInd_new))
        sensWaterHeatDemand_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_ground_hourly = numpy.zeros(len(Output_TimInd_new))
        elecDomesticDemand_hourly = numpy.zeros(len(Output_TimInd_new))
        Q_waterRecovery_hourly = numpy.zeros(len(Output_TimInd_new))
        Trural_epw = numpy.zeros(len(Output_TimInd_new))

        # Check validity of input parameters
        if self.u_star_min_MOST < 0.1:
            print('Error : Minimum friction velocity in the rural area is less than 0.1. Please check "u_star_min_MOST" in the input file')
            quit()
        if self.WindMin_MOST < 0.2:
            print('Error : Minimum wind speed in the rural area is less than 0.2. Please check "WindMin_MOST" in the input file')
            quit()
        if self.zToverz0_MOST < 0.1 or self.zToverz0_MOST > 1:
            print('Error : Thermodynamic roughness length over aerodynamic roughness length is unrealistic. Please check "zToverz0_MOST" in the input file')
            quit()
        if self.dispoverh_MOST < 0.2 or self.dispoverh_MOST > 1:
            print('Error : Displacement height over obstacle height is unrealistic. Please check "dispoverh_MOST" in the input file')
            quit()
        if self.nz*self.dz > 4*self.bldHeight or self.nz*self.dz < 2*self.bldHeight:
            print('Error : Domain height can not be higher than four times of building height or less than two times of building height. Please check "bldHeight", "nz", and "dz" in the input file ')
            quit()
        if self.nz*self.dz > 150:
            print('Error : Rural model may not be valid for this domain height. Please check "bldHeight", "nz", and "dz" in the input file ')
            quit()
        if self.simTime.dt < 60 or self.simTime.dt > 600:
            print('Error : Simulation time step is too fine or coarse. Please check "dt" in the input file')
            quit()
        if self.Cbw > 3 or self.Cbw < 0.3:
            print('Error: building width to street width ratio may be out of range. Please check "Cbw" in the input file.')
            quit()
        if self.theta_can > 90 or self.theta_can < -90:
            print('Error: Canyon orientation must be between -90 and 90. Please check "theta_can" in the input file.')
            quit()
        if self.bldHeight > 0.5*self.nz*self.dz or self.bldHeight < 3:
            print('Error: Building height is out of range. Please check "bldHeight" in the input file.')
            quit()
        if self.z0overh_MOST > 0.5 or self.z0overh_MOST < 0.05:
            print('Error: Aerodynamic roughness length over obstacle height is out of range. Please check "z0overh_MOST" in the input file.')
            quit()
        if self.zToverz0_MOST > 0.2 or self.zToverz0_MOST < 0.05:
            print('Error: Thermodynamic roughness length over Aerodynamic roughness length is out of range. Please check "zToverz0_MOST" in the input file.')
            quit()
        if self.dispoverh_MOST > 0.7 or self.dispoverh_MOST < 0.3:
            print('Error: Displacement height over obstacle height is out of range. Please check "dispoverh_MOST" in the input file.')
            quit()
        if self.WindMin_MOST > 0.7 or self.WindMin_MOST < 0.05:
            print('Error: Minimum wind for MOST is out of range. Please check "WindMin_MOST" in the input file.')
            quit()
        if self.h_obs > 10 or self.h_obs < 0.1:
            print('Error: Rural average obstacle height is out of range. Please check "h_obs" in the input file.')
            quit()
        if self.h_obs > 10 or self.h_obs < 0.1:
            print('Error: Rural average obstacle height is out of range. Please check "h_obs" in the input file.')
            quit()
        if self.BowenRatio_rural > 10 or self.BowenRatio_rural < -10:
            print('Error: Bowen ratio in the rural area is out of range. Please check "BowenRatio_rural" in the input file.')
            quit()
        if self.MinWind_rural > 0.7 or self.MinWind_rural < 0.05:
            print('Error: Minimum wind for rural energy balance is out of range. Please check "MinWind_rural" in the input file.')
            quit()
        if self.BowenRatio_tree > 10 or self.BowenRatio_tree < -10:
            print('Error: Bowen ratio for trees is out of range. Please check "BowenRatio_tree" in the input file.')
            quit()
        if self.radius_tree > self.distance_tree:
            print('Error: Radius of the tree is greater than the distance of tree from the wall. Please check tree parameters in the input file.')
            quit()
        if self.h_LAD[-1] > self.bldHeight:
            print('Error: Tree is higher than the building. Please check tree parameters in the input file.')
            quit()
        if 2*2*self.radius_tree > self.wx or 2*2*self.radius_tree > self.wy:
            print('Error: Tree does not fit in the canyon. Please check tree parameters in the input file.')
            quit()
        if self.windMin > 0.7 or self.windMin < 0.05:
            print('Error: Minimum wind for urban site is out of range. Please check "windMin" in the input file.')
            quit()
        if self.dz > 5 or self.dz < 1:
            print('Error: Vertical discretization should be between 1 and 5 m. Please check "dz" in the input file.')
            quit()

        for it in range(1,self.simTime.nt,1):# for every simulation time-step defined by uwg

              # Update water temperature (estimated)
              if self.is_near_zero(self.nSoil):
                  self.forc.deepTemp = sum(self.forcIP.temp)/float(len(self.forcIP.temp))
                  self.forc.waterTemp = sum(self.forcIP.temp)/float(len(self.forcIP.temp)) - 10.
              else:
                  self.forc.deepTemp = self.Tsoil[self.soilindex1][self.simTime.month-1] #soil temperature by depth, by month
                  self.forc.waterTemp = self.Tsoil[2][self.simTime.month-1]

              # There's probably a better way to update the weather...
              self.simTime.UpdateDate()

              self.logger.info("\n{0} m={1}, d={2}, h={3}, s={4}".format(__name__, self.simTime.month, self.simTime.day, self.simTime.secDay/3600., self.simTime.secDay))

              ##########################################################################################################
              #An attempt to interpolate hourly forcing data
              #ph is simulation time step in hours 10/3600 = 1/360
              #it is loop variables from 1 to 24*360*nDays, e.g. for 2 days it varies from 1 to 17,280
              # Calculate exact time in [hr]
              self.current_time = it * self.ph
              # Pick the closest lower hour from the forcing dataset
              self.nearest_lower_hour = int(math.ceil(self.current_time)) - 1
              #Check for last hour, as there is no closest upper hour from the forcing dataset
              if self.current_time > self.simTime.timeSim - 1:
                  self.forc.infra = self.forcIP.infra[self.nearest_lower_hour]
                  self.forc.temp = self.forcIP.temp[self.nearest_lower_hour]
                  self.forc.dif = self.forcIP.dif[self.nearest_lower_hour]
                  self.forc.dir = self.forcIP.dir[self.nearest_lower_hour]
                  self.forc.wind = max(self.forcIP.wind[self.nearest_lower_hour],self.geoParam.windMin)
                  self.forc.pres = self.forcIP.pres[self.nearest_lower_hour]
              else:
                  # Pick the closest lower hour from the forcing dataset
                  self.nearest_upper_hour = int(math.ceil(self.current_time))
                  #Read forcing data associated with these nearby hours
                  self.infra_lower = self.forcIP.infra[self.nearest_lower_hour]
                  self.infra_upper = self.forcIP.infra[self.nearest_upper_hour]

                  self.temp_lower = self.forcIP.temp[self.nearest_lower_hour]
                  self.temp_upper = self.forcIP.temp[self.nearest_upper_hour]

                  self.dif_lower = self.forcIP.dif[self.nearest_lower_hour]
                  self.dif_upper = self.forcIP.dif[self.nearest_upper_hour]

                  self.dir_lower = self.forcIP.dir[self.nearest_lower_hour]
                  self.dir_upper = self.forcIP.dir[self.nearest_upper_hour]

                  self.wind_lower = max(self.forcIP.wind[self.nearest_lower_hour],self.geoParam.windMin)
                  self.wind_upper = max(self.forcIP.wind[self.nearest_upper_hour],self.geoParam.windMin)

                  self.pres_lower = self.forcIP.pres[self.nearest_lower_hour]
                  self.pres_upper = self.forcIP.pres[self.nearest_upper_hour]
                  #Linear interpolation
                  self.forc.infra = ((self.infra_upper-self.infra_lower)/(self.nearest_upper_hour-self.nearest_lower_hour))\
                                *(self.current_time-self.nearest_lower_hour)+self.infra_lower

                  self.forc.temp = ((self.temp_upper - self.temp_lower) / (
                          self.nearest_upper_hour - self.nearest_lower_hour)) \
                                    * (self.current_time - self.nearest_lower_hour) + self.temp_lower

                  self.forc.dif = ((self.dif_upper - self.dif_lower) / (
                          self.nearest_upper_hour - self.nearest_lower_hour)) \
                                   * (self.current_time - self.nearest_lower_hour) + self.dif_lower

                  self.forc.dir = ((self.dir_upper - self.dir_lower) / (
                          self.nearest_upper_hour - self.nearest_lower_hour)) \
                                  * (self.current_time - self.nearest_lower_hour) + self.dir_lower

                  self.forc.wind = ((self.wind_upper - self.wind_lower) / (
                          self.nearest_upper_hour - self.nearest_lower_hour)) \
                                  * (self.current_time - self.nearest_lower_hour) + self.wind_lower

                  self.forc.pres = ((self.pres_upper - self.pres_lower) / (
                          self.nearest_upper_hour - self.nearest_lower_hour)) \
                                   * (self.current_time - self.nearest_lower_hour) + self.pres_lower

              ##########################################################################################################
              #Pick the nearest hour from the forcing data
              self.ceil_time_step = int(math.ceil(it * self.ph))-1  # simulation time increment raised to weather time step
                                                                    # minus one to be consistent with forcIP list index
              # Updating forcing instance
              self.forc.uDir = self.forcIP.uDir[self.ceil_time_step]          # wind direction
              self.forc.hum = self.forcIP.hum[self.ceil_time_step]            # specific humidty [kg kg^-1]
              self.forc.rHum = self.forcIP.rHum[self.ceil_time_step]          # Relative humidity [%]
              self.forc.prec = self.forcIP.prec[self.ceil_time_step]          # Precipitation [mm h^-1]
              self.UCM.canHum = copy.copy(self.forc.hum)                      # Canyon humidity (absolute) same as rural

              # ========================================================================================================
              # 1-D Model (Sec.2 Start)
              # ========================================================================================================
              # Option 1: Meili et al., 2020
              class geometry_Def():
                  pass
              geometry = geometry_Def()
              geometry.hcanyon = self.bldHeight / self.UCM.canWidth
              geometry.wcanyon = self.UCM.canWidth / self.UCM.canWidth
              geometry.htree = max(self.h_LAD) / self.UCM.canWidth
              geometry.radius_tree = self.radius_tree / self.UCM.canWidth
              geometry.distance_tree = self.distance_tree / self.UCM.canWidth

              # Surface fraction
              class FractionsGround_Def():
                  pass
              FractionsGround = FractionsGround_Def()
              FractionsGround.fveg = self.vegCover
              FractionsGround.fbare = self.soilCover
              FractionsGround.fimp = 1-(self.vegCover+self.soilCover)

              # Trees parameters
              class ParTree_Def():
                  pass
              ParTree = ParTree_Def()
              ParTree.trees = self.trees
              ParTree.ftree = self.ftree

              # Optical properties of ground
              class PropOpticalGround_Def():
                  pass
              PropOpticalGround = PropOpticalGround_Def()
              PropOpticalGround.eveg = self.eps_veg
              PropOpticalGround.ebare = self.eps_bare
              PropOpticalGround.eimp = self.eps_road
              PropOpticalGround.aveg = self.alb_veg
              PropOpticalGround.abare = self.alb_bare
              PropOpticalGround.aimp = self.alb_road

              # Optical properties of wall
              class PropOpticalWall_Def():
                  pass
              PropOpticalWall = PropOpticalWall_Def()
              PropOpticalWall.emissivity = self.eps_wall
              PropOpticalWall.albedo = self.alb_wall

              # Optical properties of tree
              class PropOpticalTree_Def():
                  pass
              PropOpticalTree = PropOpticalTree_Def()
              PropOpticalTree.emissivity = self.eps_veg
              PropOpticalTree.albedo = self.alb_veg

              # Tree parameters
              class ParVegTree_Def():
                  pass
              ParVegTree = ParVegTree_Def()
              ParVegTree.LAI = self.LAI
              ParVegTree.Kopt = 0.5

              # Meteo data
              class MeteoData_Def():
                  pass
              MeteoData = MeteoData_Def()
              MeteoData.LWR = self.forc.infra
              MeteoData.SW_dir = self.forc.dir
              MeteoData.SW_diff = self.forc.dif

              # Calculate sun position
              def SetSunVariables(Datam, DeltaGMT, Lon, Lat, t_bef, t_aft):
                  # Determine the julian day of the current time
                  days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                  nowYR = int(Datam[0])
                  nowMO = int(Datam[1])
                  nowDA = int(Datam[2])
                  nowHR = Datam[3] + Datam[4] / 60 + Datam[5] / 3600

                  if nowMO == 1:
                      jDay = nowDA
                  elif nowMO == 2:
                      jDay = days[0] + nowDA
                  else:
                      jDay = sum(days[0:(nowMO - 1)]) + nowDA
                      if nowYR % 4 == 0:
                          if nowYR % 400 == 0:
                              jDay = jDay + 1
                          elif nowYR % 100 != 0:
                              jDay = jDay + 1

                  # Compute solar declination
                  delta_S = 23.45 * numpy.pi / 180 * math.cos(2 * numpy.pi / 365 * (172 - jDay))
                  # Compute time difference between standard and local meridian
                  if Lon < 0:
                      Delta_TSL = -1 / 15. * (15 * abs(DeltaGMT) - abs(Lon))
                  else:
                      Delta_TSL = 1 / 15. * (15 * abs(DeltaGMT) - abs(Lon))

                  t = numpy.arange(nowHR - t_bef, nowHR + t_aft, 0.0166666)  ### check
                  tau_S = numpy.zeros(len(t))
                  for i in range(0, len(t)):
                      # Compute hour angle of the sun
                      if (t[i] < (12 + Delta_TSL)):
                          tau_S[i] = 15 * numpy.pi / 180 * (t[i] + 12 - Delta_TSL)
                      else:
                          tau_S[i] = 15 * numpy.pi / 180 * (t[i] - 12 - Delta_TSL)

                  # Compute solar altitude
                  Lat_rad = Lat * numpy.pi / 180
                  sinh_S = [
                      math.sin(Lat_rad) * math.sin(delta_S) + math.cos(Lat_rad) * math.cos(delta_S) * math.cos(tau_S[i])
                      for i in range(0, len(tau_S))]
                  h_S = [math.asin(sinh_S[i]) for i in range(0, len(tau_S))]
                  h_S = numpy.mean(h_S)

                  # Compute Sun's azimuth
                  zeta_S = [math.atan(-math.sin(tau_S[i]) / (
                          math.tan(delta_S) * math.cos(Lat_rad) - math.sin(Lat_rad) * math.cos(tau_S[i]))) for i in
                            range(0, len(tau_S))]

                  for i in range(0, len(t)):
                      if (tau_S[i] > 0 and tau_S[i] <= numpy.pi):
                          if (zeta_S[i] > 0.):
                              zeta_S[i] = zeta_S[i] + numpy.pi
                          else:
                              zeta_S[i] = zeta_S[i] + (2. * numpy.pi)
                      elif tau_S[i] >= numpy.pi and tau_S[i] <= 2 * numpy.pi:
                          if zeta_S[i] < 0.:
                              zeta_S[i] = zeta_S[i] + numpy.pi

                  zeta_S = numpy.mean(zeta_S)

                  # Compute sunrise time, sunset time, and total day length
                  T_sunrise = 180 / (15 * numpy.pi) * (
                          2 * numpy.pi - math.acos(-math.tan(delta_S) * math.tan(Lat_rad))) - 12
                  T_sunset = 180 / (15 * numpy.pi) * math.acos(-math.tan(delta_S) * math.tan(Lat_rad)) + 12
                  L_day = 360 / (15 * numpy.pi) * math.acos(-math.tan(delta_S) * math.tan(Lat_rad))
                  T_sunrise = numpy.real(T_sunrise)
                  T_sunset = numpy.real(T_sunset)
                  L_day = numpy.real(L_day)

                  return h_S, delta_S, zeta_S, T_sunrise, T_sunset, L_day, jDay

              # difference with Greenwich Meridian Time [h]
              DeltaGMT = self.GMT
              # longitude positive east (degrees)
              LAMBDA = self.RSM.lon
              # latitude positive north (degrees)
              phi = self.RSM.lat
              HOUR = int(self.simTime.secDay / 3600. % 24.)
              MINUTE = int(self.simTime.secDay / 60. % 60.)
              Datam = [2002, self.simTime.month, self.simTime.day, HOUR, MINUTE, 0]
              t_bef = 0.5
              t_aft = 0.5
              h_S, _a_, zeta_S, _b_, _c_, _d_, _e_ = SetSunVariables(Datam, DeltaGMT, LAMBDA, phi, t_bef, t_aft)
              # solar zenith angle
              theta_Z = numpy.pi / 2 - h_S
              if theta_Z <= -numpy.pi / 2 or theta_Z >= numpy.pi / 2:
                  theta_Z = numpy.pi / 2
              # canyon orientation (rad)
              theta_canyon = self.theta_can * numpy.pi / 180
              # difference between solar azimuth angle and canyon orientation
              theta_n = zeta_S - theta_canyon

              class SunPosition_Def():
                  pass
              SunPosition = SunPosition_Def()
              SunPosition.theta_Z = theta_Z
              SunPosition.theta_n = theta_n

              class ViewFactor_Def():
                  pass
              ViewFactor = ViewFactor_Def()
              ViewFactor.F_gs_nT = self.F_gs_nT
              ViewFactor.F_gw_nT = self.F_gw_nT
              ViewFactor.F_ww_nT = self.F_ww_nT
              ViewFactor.F_wg_nT = self.F_wg_nT
              ViewFactor.F_ws_nT = self.F_ws_nT
              ViewFactor.F_sg_nT = self.F_sg_nT
              ViewFactor.F_sw_nT = self.F_sw_nT
              ViewFactor.F_gs_T = self.F_gs_T
              ViewFactor.F_gt_T = self.F_gt_T
              ViewFactor.F_gw_T = self.F_gw_T
              ViewFactor.F_ww_T = self.F_ww_T
              ViewFactor.F_wt_T = self.F_wt_T
              ViewFactor.F_wg_T = self.F_wg_T
              ViewFactor.F_ws_T = self.F_ws_T
              ViewFactor.F_sg_T = self.F_sg_T
              ViewFactor.F_sw_T = self.F_sw_T
              ViewFactor.F_st_T = self.F_st_T
              ViewFactor.F_tg_T = self.F_tg_T
              ViewFactor.F_tw_T = self.F_tw_T
              ViewFactor.F_ts_T = self.F_ts_T
              ViewFactor.F_tt_T = self.F_tt_T

              Tveg_nonzero = [self.Tveg[jTree] for jTree in range(len(self.Tveg)) if self.Tveg[jTree] > 0]
              self.UCM.Ttree = numpy.mean(Tveg_nonzero)
              # Define surface temperatures
              TemperatureC = numpy.zeros(6)
              TemperatureC[0] = self.UCM.roadTemp
              TemperatureC[1] = 303.15
              TemperatureC[2] = 303.15
              TemperatureC[3] = self.BEM[0].wall.layerTemp[0]
              TemperatureC[4] = self.BEM[0].wall.layerTemp[0]
              TemperatureC[5] = self.UCM.Ttree

              SolarModel = RadiationFunctions()
              SWRin_t, SWRout_t, SWRabs_t, SWRabsDir_t, SWRabsDiff_t, SWREB_t = \
                  SolarModel.TotalSWRabsorbed(geometry, FractionsGround, ParTree, PropOpticalGround, PropOpticalWall,
                                              PropOpticalTree, ParVegTree, MeteoData, SunPosition, ViewFactor)
              LWRin_t, LWRout_t, LWRabs_t, LWREB_t = \
                  SolarModel.TotalLWRabsorbed(TemperatureC, geometry, MeteoData,FractionsGround,PropOpticalGround,
                                              PropOpticalWall,PropOpticalTree,ParTree, ViewFactor)

              # Update radiation term
              self.UCM.road.solRec = SWRin_t.SWRinGroundImp
              self.UCM.road.solAbs = SWRabs_t.SWRabsGroundImp
              self.UCM.SolRecRoad = SWRin_t.SWRinGroundImp
              self.UCM.SolRecWall = SWRin_t.SWRinWallSun
              self.UCM.road.infra = LWRabs_t.LWRabsGroundImp
              for jBEM in range(len(self.BEM)):
                  self.BEM[jBEM].wall.solRec = SWRin_t.SWRinWallSun
                  self.BEM[jBEM].wall.solAbs = SWRabs_t.SWRabsWallSun
                  self.BEM[jBEM].wall.infra = LWRabs_t.LWRabsWallSun

              # Calculate outgoing and net longwave radiation in the rural area  [W m^-2]
              # Outgoing longwave radiation [W m^-2]
              self.L_rural_emt = self.eps_rural * self.SIGMA * self.rural.layerTemp[0] ** 4
              # Net longwave radiation at the rural surface [W m^-2]
              self.rural.infra = self.eps_rural *self.forc.infra - self.L_rural_emt

              # Calculate incoming and net shortwave in the rural
              SDir_rural = max(math.cos(theta_Z) * self.forc.dir, 0.0)
              # Winter: no vegetation
              if self.simTime.month < self.geoParam.vegStart or self.simTime.month > self.geoParam.vegEnd:
                  self.S_rural = (1 - self.alb_rural) * (SDir_rural + self.forc.dif)
              # Summer: effect of vegetation is considered
              else:
                  self.S_rural = ((1 - self.rurVegCover) * (1 - self.alb_rural) + self.rurVegCover * (1 - self.alb_veg)) * (SDir_rural + self.forc.dif)

              self.rural.solRec = SDir_rural + self.forc.dif
              self.rural.solAbs = self.S_rural

              # Calculate incoming and net longwave radiation of the roof [W m^-2]
              # Outgoing longwave radiation [W m^-2]
              self.L_roof_emt = self.eps_roof * self.SIGMA * self.UCM.roofTemp ** 4
              # Net longwave radiation at the rural surface [W m^-2]
              self.L_roof_abs = self.forc.infra - self.L_roof_emt

              # Calculate incoming and net shortwave in the roof
              SDir_roof = max(math.cos(theta_Z) * self.forc.dir, 0.0)
              self.S_roof_abs = (1 - self.alb_roof) * (SDir_roof + self.forc.dif)

              for jBEM in xrange(len(self.BEM)):
                  # Solar received by roof [W m^-2]
                  self.BEM[jBEM].roof.solRec = SDir_roof + self.forc.dif
                  # Solar absorbed by roof [W m^-2]
                  self.BEM[jBEM].roof.solAbs = self.S_roof_abs
                  # Update net longwave radiation of the roof [W m^-2]
                  self.BEM[jBEM].roof.infra = self.L_roof_abs
              S_roof_out = self.alb_roof*(SDir_roof + self.forc.dif)
              # Calculate trees temperature and sensible/latent heat fluxes
              TreeEB = Tree_EB()
              self.Tveg, Q_HV, Q_LV = \
                  TreeEB.TreeCal(self.th,self.vx,self.vy,SWRabs_t.SWRabsTree,self.R,self.CP,self.omega,self.leaf_width,
                                 self.nz,self.dz,self.h_LAD,self.BowenRatio_tree)
              self.UCM.treeSensHeat = numpy.mean(Q_HV)
              self.UCM.treeLatHeat = numpy.mean(Q_LV)
              # ========================================================================================================
              # 1-D Model (Sec.2 End)
              # ========================================================================================================

              # Update building & traffic schedule
              # Assign day type (1 = weekday, 2 = sat, 3 = sun/other)
              if self.is_near_zero(self.simTime.julian % 7):
                  self.dayType = 3                                   # Sunday
              elif self.is_near_zero(self.simTime.julian % 7 - 6.):
                  self.dayType = 2                                   # Saturday
              else:
                  self.dayType = 1                                   # Weekday

              # Update anthropogenic heat load for each hour [W m^-2]
              self.UCM.sensAnthrop = self.sensAnth * (self.SchTraffic[self.dayType-1][self.simTime.hourDay])

              # Update the energy components for building types defined in initialize.uwg
              for i in xrange(len(self.BEM)):
                  # Set point temperature [K]
                  # Add from temperature schedule for cooling
                  self.BEM[i].building.coolSetpointDay   = self.Sch[i].Cool[self.dayType-1][self.simTime.hourDay] + 273.15
                  self.BEM[i].building.coolSetpointNight = self.BEM[i].building.coolSetpointDay
                  # Add from temperature schedule for heating
                  self.BEM[i].building.heatSetpointDay   = self.Sch[i].Heat[self.dayType-1][self.simTime.hourDay] + 273.15
                  self.BEM[i].building.heatSetpointNight = self.BEM[i].building.heatSetpointDay

                  # Internal Heat Load Schedule per unit floor area [W m^-2]
                  # Electricity consumption per unit floor area [W m^-2] = max for electrical plug process * electricity fraction for the day
                  self.BEM[i].Elec  = self.Sch[i].Qelec * self.Sch[i].Elec[self.dayType-1][self.simTime.hourDay]
                  # Lighting per unit floor area [W m^-2] = max for light * light fraction for the day
                  self.BEM[i].Light = self.Sch[i].Qlight * self.Sch[i].Light[self.dayType-1][self.simTime.hourDay]
                  # Number of occupants x occ fraction for day
                  self.BEM[i].Nocc  = self.Sch[i].Nocc * self.Sch[i].Occ[self.dayType-1][self.simTime.hourDay]
                  # Sensible Q occupant * fraction occupant sensible Q * number of occupants
                  self.BEM[i].Qocc  = self.sensOcc * (1 - self.LatFOcc) * self.BEM[i].Nocc

                  # SWH and ventilation schedule
                  # Solar water heating per unit floor area [W m^-2] = Peak Service Hot Water per unit floor [kg hr^-1 m^-2] * SWH fraction for the day
                  self.BEM[i].SWH = self.Sch[i].Vswh * self.Sch[i].SWH[self.dayType-1][self.simTime.hourDay]
                  # Ventilation rate per unit floor area [m^3 s^-1 m^-2]
                  self.BEM[i].building.vent = self.Sch[i].Vent
                  # Gas consumption per unit floor area [W m^-2] = max for gas * Gas fraction for the day
                  self.BEM[i].Gas = self.Sch[i].Qgas * self.Sch[i].Gas[self.dayType-1][self.simTime.hourDay]

                  # This is quite messy, should update
                  # Update internal heat and corresponding fractional loads per unit floor area [W m^-2]
                  intHeat = self.BEM[i].Light + self.BEM[i].Elec + self.BEM[i].Qocc
                  self.BEM[i].building.intHeatDay = intHeat
                  self.BEM[i].building.intHeatNight = intHeat
                  # Fraction of radiant heat from light and equipment of whole internal heat per unit floor area [W m^-2]
                  self.BEM[i].building.intHeatFRad = (self.RadFLight * self.BEM[i].Light + self.RadFEquip * self.BEM[i].Elec) / intHeat
                  # fraction of latent heat (from occupants) of whole internal heat per unit floor area [W m^-2]
                  self.BEM[i].building.intHeatFLat = self.LatFOcc * self.sensOcc * self.BEM[i].Nocc/intHeat

                  # Update envelope temperature layers [K]
                  self.BEM[i].T_wallex = self.BEM[i].wall.layerTemp[0]   # Wall temperature exposed to outdoor environment [K]
                  self.BEM[i].T_wallin = self.BEM[i].wall.layerTemp[-1]  # Wall temperature exposed to indoor environment [K]
                  self.BEM[i].T_roofex = self.BEM[i].roof.layerTemp[0]   # Roof temperature exposed to outdoor environment [K]
                  self.BEM[i].T_roofin = self.BEM[i].roof.layerTemp[-1]  # Roof temperature exposed to indoor environment [K]

              # Update heat fluxes [W m^-2] and surface temperature [K] in rural area
              self.rural.SurfFlux(self.forc, self.geoParam, self.simTime, self.forc.hum, self.forc.temp, self.forc.wind,
                                  2., 0.,self.h_obs,self.h_temp,self.BowenRatio_rural,self.MinWind_rural)

              # Update vertical diffusion model (VDM): Calculate temperature profile, wind speed profile and density profile in rural area
              self.RSM.MOST(self.rural)

              # Update UWG wind speed within the canyon by taking average of velocity profiles within the canyon
              WindUrban = numpy.sqrt(numpy.mean(self.vx[0:self.nz_u])**2+numpy.mean(self.vy[0:self.nz_u])**2)
              WindRoof = numpy.sqrt((self.vx[self.nz_u])**2 + (self.vy[self.nz_u])**2)

              # Calculate urban heat fluxes
              self.UCM, self.BEM, dens = urbflux(self.UCM,self.BEM, self.forc, self.geoParam, self.simTime, self.RSM, WindUrban, self.bx, self.by,
                                                 self.beta_st, self.A_st, self.U_st, self.FR_st, self.tau_alpha_e_st,
                                                 self.eta_he_st, self.V_bites, self.c_bites,
                                                 self.m_dot_st_f, self.c_st_f, self.m_dot_he_st,
                                                 theta_Z, zeta_S, self.th[self.nz_u], self.Adv_ene_heat_mode,
                                                 self.beta_pv, self.A_pv, self.eta_pv, self.COP_hp_min, self.COP_hp_max, self.T_hp_min, self.T_hp_max,
                                                 self.A_wt, self.eta_wt, self.S_wt_min, self.S_wt_max, WindRoof,
                                                 self.V_pcm, self.l_pcm, self.T_melt, self.rural)

              self.UCM.UCModel(self.BEM, self.RSM.T_rural[-1], self.forc, self.geoParam)

              # ========================================================================================================
              # 1-D Model (Sec.3 Start)
              # ========================================================================================================
              # Calculate density profile of density [kg m^-3]
              rho_prof = numpy.zeros(self.nz)
              for i_rho in range(0,self.nz):
                  # a constant density lapse rate of - 0.000133 [kg m-3 m-1]
                  rho_prof[i_rho] = self.UCM.rhoCan-0.000133*(self.z[i_rho]-0)

              # Update total sensible waste heat to canyon per unit building footprint area [W m^-2]
              SensHt_HVAC = 0
              for iBEM_sensheat in range(0, len(self.BEM)):
                  SensHt_HVAC += self.BEM[iBEM_sensheat].building.sensWaste

              # Calculate potential temperature, wind speed, specific humidity, and turbulent kinetic energy profiles in the urban area
              self.ColModelParam = ColModel(self.UCM.wallTemp,self.UCM.roofTemp,self.UCM.roadTemp,self.RSM.T_rural[self.nz-1],self.RSM.q_rural[self.nz-1],self.RSM.q_rural[1],
                                            self.forc.wind,self.forc.uDir, self.vx, self.vy, self.tke, self.th, self.qn,self.nz, self.Ck, self.dlk,
                                            self.nz_u, self.dz, self.simTime.dt, self.vol,self.road.vegCoverage, self.lambdap, self.lambdaf,
                                            self.bldHeight, self.CP,self.th_ref, self.Cdrag, self.pb, self.ss, self.prandtl,self.schmidt, self.G,
                                            self.Ceps, self.dls,self.sf, rho_prof,self.h_LAD,self.f_LAD,
                                            SensHt_HVAC,self.theta_can,self.HVAC_street_frac,self.HVAC_atm_frac,self.RSM.u_star,
                                            self.Ri_b_cr,self.Ck_stable,self.Ck_unstable,self.windMin)

              self.vx, self.vy, self.tke, self.th, self.qn = \
                  self.ColModelParam.ColumnModelCal(self.z0_road,self.z0_roof,self.R,self.cdmin,self.C_dpdx,self.leaf_width,
                                                    self.omega,self.omega_drag,self.LV,self.Cmu,self.Tveg)

              # ========================================================================================================
              # 1-D Model (Sec.3 End)
              # ========================================================================================================

              self.logger.info("dbT = {}".format(self.UCM.canTemp-273.15))
              if n > 0:
                  logging.info("dpT = {}".format(self.UCM.Tdp))
                  logging.info("RH  = {}".format(self.UCM.canRHum))

              if self.is_near_zero(self.simTime.secDay % self.simTime.timePrint) and n < self.N:

                  self.logger.info("{0} ----sim time step = {1}----\n\n".format(__name__, n))

                  self.WeatherData[n] = copy.copy(self.forc)
                  _Tdb, _w, self.UCM.canRHum, _h, self.UCM.Tdp, _v = psychrometrics(self.UCM.canTemp, self.UCM.canHum, self.forc.pres)

                  self.UCMData[n] = copy.copy(self.UCM)
                  self.RSMData[n] = copy.copy(self.RSM)

                  self.logger.info("dbT = {}".format(self.UCMData[n].canTemp-273.15))
                  self.logger.info("dpT = {}".format(self.UCMData[n].Tdp))
                  self.logger.info("RH  = {}".format(self.UCMData[n].canRHum))

                  # Print progress bar
                  sim_it = round((it/float(self.simTime.nt))*100.0,1)
                  ProgressBar.print_progress(sim_it, 100.0, prefix = "Progress:", bar_length = 25)

                  n += 1

              # Save simulation results every hour
              if it == Output_TimInd_new[iO_new]:
                  # output data
                  # Save profiles
                  U_hourly[:][iO_new] = self.vx
                  V_hourly[:][iO_new] = self.vy
                  Tu_hourly[:][iO_new] = self.th
                  Tr_hourly[:][iO_new] = self.RSM.T_rural
                  q_hourly[:][iO_new] = self.qn
                  TKE_hourly[:][iO_new] = self.tke
                  # Save building energy terms
                  sensWaste_hourly[iO_new] = sum([self.BEM[iBEM].building.sensWaste for iBEM in range(0, len(self.BEM))])
                  dehumDemand_hourly[iO_new] = sum([self.BEM[iBEM].building.dehumDemand for iBEM in range(0, len(self.BEM))])
                  QWater_hourly[iO_new] = sum([self.BEM[iBEM].building.QWater for iBEM in range(0, len(self.BEM))])
                  QGas_hourly[iO_new] = sum([self.BEM[iBEM].building.QGas for iBEM in range(0, len(self.BEM))])
                  sensCoolDemand_hourly[iO_new] = sum([self.BEM[iBEM].building.sensCoolDemand for iBEM in range(0, len(self.BEM))])
                  coolConsump_hourly[iO_new] = sum([self.BEM[iBEM].building.coolConsump for iBEM in range(0, len(self.BEM))])
                  sensHeatDemand_hourly[iO_new] = sum([self.BEM[iBEM].building.sensHeatDemand for iBEM in range(0, len(self.BEM))])
                  heatConsump_hourly[iO_new] = sum([self.BEM[iBEM].building.heatConsump for iBEM in range(0, len(self.BEM))])
                  Q_st_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_st for iBEM in range(0, len(self.BEM))])
                  Q_he_st_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_he_st for iBEM in range(0, len(self.BEM))])
                  Q_bites_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_bites for iBEM in range(0, len(self.BEM))])
                  Q_hp_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_hp for iBEM in range(0, len(self.BEM))])
                  Q_recovery_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_recovery for iBEM in range(0, len(self.BEM))])
                  W_hp_hourly[iO_new] = sum([self.BEM[iBEM].building.W_hp for iBEM in range(0, len(self.BEM))])
                  W_pv_hourly[iO_new] = sum([self.BEM[iBEM].building.W_pv for iBEM in range(0, len(self.BEM))])
                  COP_hp_hourly[iO_new] = sum([self.BEM[iBEM].building.COP_hp for iBEM in range(0, len(self.BEM))])
                  indoorTemp_hourly[iO_new] = sum([self.BEM[iBEM].building.indoorTemp for iBEM in range(0, len(self.BEM))])
                  T_st_f_i_hourly[iO_new] = sum([self.BEM[iBEM].building.T_st_f_i for iBEM in range(0, len(self.BEM))])
                  T_st_f_o_hourly[iO_new] = sum([self.BEM[iBEM].building.T_st_f_o for iBEM in range(0, len(self.BEM))])
                  T_he_st_i_hourly[iO_new] = sum([self.BEM[iBEM].building.T_he_st_i for iBEM in range(0, len(self.BEM))])
                  T_he_st_o_hourly[iO_new] = sum([self.BEM[iBEM].building.T_he_st_o for iBEM in range(0, len(self.BEM))])
                  T_bites_hourly[iO_new] = sum([self.BEM[iBEM].building.T_bites for iBEM in range(0, len(self.BEM))])
                  W_wt_hourly[iO_new] = sum([self.BEM[iBEM].building.W_wt for iBEM in range(0, len(self.BEM))])
                  f_pcm_hourly[iO_new] = sum([self.BEM[iBEM].building.f_pcm for iBEM in range(0, len(self.BEM))])
                  Q_waterSaved_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_waterSaved for iBEM in range(0, len(self.BEM))])
                  sensWaterHeatDemand_hourly[iO_new] = sum([self.BEM[iBEM].building.sensWaterHeatDemand for iBEM in range(0, len(self.BEM))])
                  Q_ground_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_ground for iBEM in range(0, len(self.BEM))])
                  elecDomesticDemand_hourly[iO_new] = sum([self.BEM[iBEM].building.elecDomesticDemand for iBEM in range(0, len(self.BEM))])
                  Q_waterRecovery_hourly[iO_new] = sum([self.BEM[iBEM].building.Q_waterRecovery for iBEM in range(0, len(self.BEM))])
                  Trural_epw[iO_new] = self.forc.temp

                  if iO_new < len(Output_TimInd_new)-1:
                      iO_new += 1

        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ######################################## Generate Output files #################################################

        # Generate output text file for U
        Header0 = "#0:z [m] "
        Values_format0 = "%f "
        for io in range(0, len(Output_TimInd_new)):
            Header = Header0 + str(io + 1) + ":U [m s^-1] "
            Header0 = Header
            Values_format = Values_format0 + " %f "
            Values_format0 = Values_format

        Values_format_all = Values_format + str('\n')
        ProfilesFilename_U = "Output/U_profiles_hourly.txt"
        outputFileProf_U = open(ProfilesFilename_U, "w")
        outputFileProf_U.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFileProf_U.write("# Hourly vertical profile of wind speed cross canyon (x direction)  \n")
        outputFileProf_U.write(Header + "\n")
        for i in range(0, self.nz):
            Values = [U_hourly[j][i] for j in range(0,len(Output_TimInd_new))]
            Values.insert(0,self.z[i]+self.dz/2.)
            outputFileProf_U.write(Values_format_all
                                   % (tuple(Values)))
        outputFileProf_U.close()

        # Generate output text file for V
        Header0 = "#0:z [m] "
        Values_format0 = "%f "
        for io in range(0, len(Output_TimInd_new)):
            Header = Header0 + str(io + 1) + ":V [m s^-1] "
            Header0 = Header
            Values_format = Values_format0 + " %f "
            Values_format0 = Values_format

        Values_format_all = Values_format + str('\n')
        ProfilesFilename_V = "Output/V_profiles_hourly.txt"
        outputFileProf_V = open(ProfilesFilename_V, "w")
        outputFileProf_V.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFileProf_V.write("# Hourly vertical profile of wind speed along canyon (y direction)  \n")
        outputFileProf_V.write(Header + "\n")
        for i in range(0, self.nz):
            Values = [V_hourly[j][i] for j in range(0, len(Output_TimInd_new))]
            Values.insert(0, self.z[i]+self.dz/2.)
            outputFileProf_V.write(Values_format_all
                                   % (tuple(Values)))
        outputFileProf_V.close()

        # Generate output text file for Tr
        Header0 = "#0:z [m] "
        Values_format0 = "%f "
        for io in range(0, len(Output_TimInd_new)):
            Header = Header0 + str(io + 1) + ":Tr [K] "
            Header0 = Header
            Values_format = Values_format0 + " %f "
            Values_format0 = Values_format

        Values_format_all = Values_format + str('\n')
        ProfilesFilename_Tr = "Output/Tr_profiles_hourly.txt"
        outputFileProf_Tr = open(ProfilesFilename_Tr, "w")
        outputFileProf_Tr.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFileProf_Tr.write("# Hourly vertical profile of potential temperature in the rural area \n")
        outputFileProf_Tr.write(Header + "\n")
        for i in range(0, self.nz):
            Values = [Tr_hourly[j][i] for j in range(0, len(Output_TimInd_new))]
            Values.insert(0, self.z[i]+self.dz/2.)
            outputFileProf_Tr.write(Values_format_all
                                   % (tuple(Values)))
        outputFileProf_Tr.close()

        # Generate output text file for Tu
        Header0 = "#0:z [m] "
        Values_format0 = "%f "
        for io in range(0, len(Output_TimInd_new)):
            Header = Header0 + str(io + 1) + ":Tu [K] "
            Header0 = Header
            Values_format = Values_format0 + " %f "
            Values_format0 = Values_format

        Values_format_all = Values_format + str('\n')
        ProfilesFilename_Tu = "Output/Tu_profiles_hourly.txt"
        outputFileProf_Tu = open(ProfilesFilename_Tu, "w")
        outputFileProf_Tu.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFileProf_Tu.write("# Hourly vertical profile of potential temperature in the urban area \n")
        outputFileProf_Tu.write(Header + "\n")
        for i in range(0, self.nz):
            Values = [Tu_hourly[j][i] for j in range(0, len(Output_TimInd_new))]
            Values.insert(0, self.z[i]+self.dz/2.)
            outputFileProf_Tu.write(Values_format_all
                                   % (tuple(Values)))
        outputFileProf_Tu.close()

        # Generate output text file for qn
        Header0 = "#0:z [m] "
        Values_format0 = "%f "
        for io in range(0, len(Output_TimInd_new)):
            Header = Header0 + str(io + 1) + ":q [Kg Kg^-1] "
            Header0 = Header
            Values_format = Values_format0 + " %f "
            Values_format0 = Values_format

        Values_format_all = Values_format + str('\n')
        ProfilesFilename_q = "Output/q_profiles_hourly.txt"
        outputFileProf_q = open(ProfilesFilename_q, "w")
        outputFileProf_q.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFileProf_q.write("# Hourly vertical profile of specific humidity in the urban area \n")
        outputFileProf_q.write(Header + "\n")
        for i in range(0, self.nz):
            Values = [q_hourly[j][i] for j in range(0, len(Output_TimInd_new))]
            Values.insert(0, self.z[i]+self.dz/2.)
            outputFileProf_q.write(Values_format_all
                                   % (tuple(Values)))
        outputFileProf_q.close()

        # Generate output text file for TKE
        Header0 = "#0:z [m] "
        Values_format0 = "%f "
        for io in range(0, len(Output_TimInd_new)):
            Header = Header0 + str(io + 1) + ":TKE [m^2 s^-2] "
            Header0 = Header
            Values_format = Values_format0 + " %f "
            Values_format0 = Values_format

        Values_format_all = Values_format + str('\n')
        ProfilesFilename_TKE = "Output/TKE_profiles_hourly.txt"
        outputFileProf_TKE = open(ProfilesFilename_TKE, "w")
        outputFileProf_TKE.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFileProf_TKE.write("# Hourly vertical profile of turbulence kinetic energy in the urban area \n")
        outputFileProf_TKE.write(Header + "\n")
        for i in range(0, self.nz):
            Values = [TKE_hourly[j][i] for j in range(0, len(Output_TimInd_new))]
            Values.insert(0, self.z[i]+self.dz/2.)
            outputFileProf_TKE.write(Values_format_all
                                   % (tuple(Values)))
        outputFileProf_TKE.close()

        # Generate output text file for building energy fluxes
        timeseriesFilename = "Output/BEM_hourly.txt"
        outputFile_BEM = open(timeseriesFilename, "w")
        outputFile_BEM.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFile_BEM.write("# Hourly building energy data \n")
        outputFile_BEM.write("# 0:time [hr] 1:sensWaste [W m^-2] 2:dehumDemand [W m^-2] 3:QWater [W m^-2] 4:QGas [W m^-2] "
                             "5: sensCoolDemand [W m^-2] 6:coolConsump [W m^-2] 7:sensHeatDemand [W m^-2] 8: heatConsump [W m^-2] "
                             "9: Q_st [W m^-2] 10: Q_he_st [W m^-2] 11: Q_bites [W m^-2] 12: Q_hp [W m^-2] 13: Q_recovery [W m^-2]"
                             "14: W_hp [W m^-2] 15: W_pv [W m^-2] 16: COP_hp"
                             "17: indoorTemp [K] 18: T_st_f_i [K] 19: T_st_f_o [K] 20: T_he_st_i [K] 21: T_he_st_o [K] 22: T_bites [K]"
                             "23: W_wt [W m^-2] 24: f_pcm 25: Q_waterSaved [W m^-2] 26: sensWaterHeatDemand [W m^-2]"
                             "27: Q_ground [W m^-2] 28: elecDomesticDemand [W m^-2] 29: Q_waterRecovery [W m^-2] \n")
        for i in range(0,len(Output_TimInd_new)):
            outputFile_BEM.write("%i %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n"
                                 % (i,sensWaste_hourly[i],dehumDemand_hourly[i],QWater_hourly[i],QGas_hourly[i],sensCoolDemand_hourly[i],
                                    coolConsump_hourly[i],sensHeatDemand_hourly[i], heatConsump_hourly[i],
                                    Q_st_hourly[i],Q_he_st_hourly[i],Q_bites_hourly[i], Q_hp_hourly[i], Q_recovery_hourly[i],
                                    W_hp_hourly[i], W_pv_hourly[i], COP_hp_hourly[i],
                                    indoorTemp_hourly[i],T_st_f_i_hourly[i],T_st_f_o_hourly[i],
                                    T_he_st_i_hourly[i],T_he_st_o_hourly[i],T_bites_hourly[i],
                                    W_wt_hourly[i], f_pcm_hourly[i], Q_waterSaved_hourly[i],sensWaterHeatDemand_hourly[i],
                                    Q_ground_hourly[i], elecDomesticDemand_hourly[i], Q_waterRecovery_hourly[i]))
        outputFile_BEM.close()

        # T_epw
        timeseriesFilename = "Output/Tepw_hourly.txt"
        outputFile_Tepw = open(timeseriesFilename, "w")
        outputFile_Tepw.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
        outputFile_Tepw.write("# Hourly T_epw \n")
        outputFile_Tepw.write("# 0:time [hr] 1:T_epw [K]\n")
        for i in range(0, len(Output_TimInd_new)):
            outputFile_Tepw.write("%i %f \n"
                                 % (i, Trural_epw[i]))
        outputFile_Tepw.close()

    def write_epw(self):
        """ Section 8 - Writing new EPW file
        """
        epw_prec = self.epw_precision # precision of epw file input

        for iJ in xrange(len(self.UCMData)):
            # [iJ+self.simTime.timeInitial-8] = increments along every weather timestep in epw
            # [6 to 21]                       = column data of epw
            self.epwinput[iJ+self.simTime.timeInitial-8][6] = "{0:.{1}f}".format(self.UCMData[iJ].canTemp - 273.15, epw_prec) # dry bulb temperature  [C]
            self.epwinput[iJ+self.simTime.timeInitial-8][7] = "{0:.{1}f}".format(self.UCMData[iJ].Tdp, epw_prec)              # dew point temperature [C]
            self.epwinput[iJ+self.simTime.timeInitial-8][8] = "{0:.{1}f}".format(self.UCMData[iJ].canRHum, epw_prec)          # relative humidity     [%]
            self.epwinput[iJ+self.simTime.timeInitial-8][21] = "{0:.{1}f}".format(self.WeatherData[iJ].wind, epw_prec)        # wind speed [m s^-1]

        # Writing new EPW file
        epw_new_id = open(self.newPathName, "w")

        for i in xrange(8):
            new_epw_line = '{}\r\n'.format(reduce(lambda x,y: x+","+y, self._header[i]))
            epw_new_id.write(new_epw_line)

        for i in xrange(len(self.epwinput)):
            printme = ""
            for ei in xrange(34):
                printme += "{}".format(self.epwinput[i][ei]) + ','
            printme = printme + "{}".format(self.epwinput[i][ei])
            new_epw_line = "{0}\r\n".format(printme)
            epw_new_id.write(new_epw_line)

        epw_new_id.close()

        print "\nNew climate file '{}' is generated at {}.".format(self.destinationFileName, self.destinationDir)

    def run(self):

        # run main class methods
        self.read_epw()
        self.set_input()
        self.instantiate_input()
        self.hvac_autosize()
        self.simulate()
        self.write_epw()


def procMat(materials,max_thickness,min_thickness):
    """ Processes material layer so that a material with single
    layer thickness is divided into two and material layer that is too
    thick is subdivided
    """
    newmat = []
    newthickness = []
    k = materials.layerThermalCond
    Vhc = materials.layerVolHeat

    if len(materials.layerThickness) > 1:

        for j in xrange(len(materials.layerThickness)):
            # Break up each layer that's more than max thickness (0.05m)
            if materials.layerThickness[j] > max_thickness:
                nlayers = math.ceil(materials.layerThickness[j]/float(max_thickness))
                for i in xrange(int(nlayers)):
                    newmat.append(Material(k[j],Vhc[j],name=materials._name))
                    newthickness.append(materials.layerThickness[j]/float(nlayers))
            # Material that's less then min_thickness is not added.
            elif materials.layerThickness[j] < min_thickness:
                print "WARNING: Material '{}' layer found too thin (<{:.2f}cm), ignored.".format(materials._name, min_thickness*100)
            else:
                newmat.append(Material(k[j],Vhc[j],name=materials._name))
                newthickness.append(materials.layerThickness[j])

    else:

        # Divide single layer into two (UWG assumes at least 2 layers)
        if materials.layerThickness[0] > max_thickness:
            nlayers = math.ceil(materials.layerThickness[0]/float(max_thickness))
            for i in xrange(int(nlayers)):
                newmat.append(Material(k[0],Vhc[0],name=materials._name))
                newthickness.append(materials.layerThickness[0]/float(nlayers))
        # Material should be at least 1cm thick, so if we're here,
        # should give warning and stop. Only warning given for now.
        elif materials.layerThickness[0] < min_thickness*2:
            newthickness = [min_thickness/2., min_thickness/2.]
            newmat = [Material(k[0],Vhc[0],name=materials._name), Material(k[0],Vhc[0],name=materials._name)]
            print "WARNING: a thin (<2cm) single material '{}' layer found. May cause error.".format(materials._name)
        else:
            newthickness = [materials.layerThickness[0]/2., materials.layerThickness[0]/2.]
            newmat = [Material(k[0],Vhc[0],name=materials._name), Material(k[0],Vhc[0],name=materials._name)]
    return newmat, newthickness
