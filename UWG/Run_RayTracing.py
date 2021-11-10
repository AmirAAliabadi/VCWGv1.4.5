from RayTracing import RayTracingCal
import numpy
import time
import math
import copy

"""
Calculate view factors
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: November 2020
Originally developed by Naika Meili
"""

# OPTION_RAY = 0 does overwrite file 'ViewFactor_BASEL.txt' with new view factors.
# This will run both analytical and ray-tracing models.
# OPTION_RAY = 1 does not overwrite file 'ViewFactor_BASEL.txt'.
# This will only run the analytical model and copy ray-tracing results from the file.
OPTION_RAY = 0
ViewFactor_file = 'ViewFactor_Guelph.txt'
# Make sure the following parameters are the same as the initialization file.
Width_canyon = 23   # [m]
Height_canyon = 6   # [m]
Radius_tree = 1.5   # [m]
Distance_tree = 2.2 # Distance of tree to wall [m]
Height_tree = 3.5   # Tree height from ground to middle of crown [m]

class Geometry_m_Def():
    pass
Geometry_m = Geometry_m_Def()
Geometry_m.Height_canyon = Height_canyon
Geometry_m.Width_canyon = Width_canyon

class geometry_Def():
    pass
geometry = geometry_Def()
geometry.radius_tree = Radius_tree / Width_canyon
geometry.htree = Height_tree / Width_canyon
geometry.distance_tree = Distance_tree / Width_canyon


class Person_Def():
    pass
Person = Person_Def()
Person.PositionPx = Geometry_m.Width_canyon / 2
Person.PositionPz = 1.1
Person.PersonWidth = 0.03
Person.PersonHeight = 0.11
Person.HeightWind = 1.1

class ParTree_Def():
    pass
ParTree = ParTree_Def()
ParTree.trees = 1


# Calculate view factors
RadFun = RayTracingCal()
ViewFactor, ViewFactorPoint = RadFun.VFUrbanCanyon(OPTION_RAY,numpy.NaN,Geometry_m,geometry,Person,ParTree,ViewFactor_file)

Symbol = ['F_gs_nT', 'F_gw_nT', 'F_ww_nT', 'F_wg_nT', 'F_ws_nT', 'F_sg_nT', 'F_sw_nT' , 'F_gs_T', 'F_gt_T', 'F_gw_T', 'F_ww_T', 'F_wt_T', 'F_wg_T',
          'F_ws_T', 'F_sg_T', 'F_sw_T' , 'F_st_T' , 'F_tg_T', 'F_tw_T' , 'F_ts_T', 'F_tt_T']
VF_values = [ViewFactor.F_gs_nT,ViewFactor.F_gw_nT,ViewFactor.F_ww_nT,ViewFactor.F_wg_nT,ViewFactor.F_ws_nT,ViewFactor.F_sg_nT,ViewFactor.F_sw_nT,
             ViewFactor.F_gs_T,ViewFactor.F_gt_T,ViewFactor.F_gw_T,ViewFactor.F_ww_T,ViewFactor.F_wt_T,ViewFactor.F_wg_T,ViewFactor.F_ws_T,
             ViewFactor.F_sg_T,ViewFactor.F_sw_T,ViewFactor.F_st_T,ViewFactor.F_tg_T,ViewFactor.F_tw_T,ViewFactor.F_ts_T,ViewFactor.F_tt_T]
outputFile_VF = open(ViewFactor_file, "w")
outputFile_VF.write("#### \t Vertical City Weather Generator (VCWG)  \t #### \n")
outputFile_VF.write("# View Factors \n")
for i in range(21):
    outputFile_VF.write("%s,%f \n" % (Symbol[i], VF_values[i]))
outputFile_VF.close()
