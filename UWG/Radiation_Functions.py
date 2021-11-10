import os
import numpy
import math
from pprint import pprint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import copy

'''
Radiation Functions:
Developed by Mohsen Moradi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Originally developed by Naika Meili
Last update: January 2021
'''

class RadiationFunctions(object):

    def TotalLWRabsorbed(self,TemperatureC,geometry,MeteoData,FractionsGround,PropOpticalGround,PropOpticalWall,
                         PropOpticalTree,ParTree,ViewFactor):

        Tgrimp = TemperatureC[0]
        Tgbare = TemperatureC[1]
        Tgveg = TemperatureC[2]
        Twsun = TemperatureC[3]
        Twshade = TemperatureC[4]
        Ttree = TemperatureC[5]
        h_can = geometry.hcanyon
        w_can = geometry.wcanyon
        r_tree = geometry.radius_tree
        fgveg = FractionsGround.fveg
        fgbare = FractionsGround.fbare
        fgimp = FractionsGround.fimp
        LWR = MeteoData.LWR
        ew = PropOpticalWall.emissivity
        et = PropOpticalTree.emissivity
        egveg = PropOpticalGround.eveg
        egbare = PropOpticalGround.ebare
        egimp = PropOpticalGround.eimp
        trees = ParTree.trees
        ftree = ParTree.ftree

        if trees == 1:
            # Longwave radiation absorbed without trees
            LWRin_nT, LWRout_nT, LWRabs_nT, LWREB_nT = self.LWRabsorbedNoTree(h_can, w_can, LWR, fgveg, fgbare, fgimp,
                                                                              ew, egveg, egbare, egimp, Tgrimp, Tgbare,
                                                                              Tgveg, Twsun, Twshade, ViewFactor)
            # Longwave radiation absorbed with trees
            LWRin_T, LWRout_T, LWRabs_T, LWREB_T = self.LWRabsorbedWithTrees(h_can, w_can, r_tree, LWR, fgveg, fgbare,
                                                                             fgimp, ew, et, egveg, egbare, egimp, Tgrimp,
                                                                             Tgbare, Tgveg, Twsun, Twshade, Ttree, ViewFactor)

            class LWRin_t_Def():
                pass
            LWRin_t = LWRin_t_Def()
            LWRin_t.LWRinGroundImp = ftree*LWRin_T.LWRinGroundImp + (1-ftree)*LWRin_nT.LWRinGroundImp
            LWRin_t.LWRinGroundBare = ftree*LWRin_T.LWRinGroundBare + (1-ftree)*LWRin_nT.LWRinGroundBare
            LWRin_t.LWRinGroundVeg = ftree*LWRin_T.LWRinGroundVeg + (1-ftree)*LWRin_nT.LWRinGroundVeg
            LWRin_t.LWRinTree = ftree*LWRin_T.LWRinTree + (1-ftree)*LWRin_nT.LWRinTree
            LWRin_t.LWRinWallSun = ftree*LWRin_T.LWRinWallSun + (1-ftree)*LWRin_nT.LWRinWallSun
            LWRin_t.LWRinWallShade = ftree*LWRin_T.LWRinWallShade + (1-ftree)*LWRin_nT.LWRinWallShade
            LWRin_t.LWRinTotalGround = ftree*LWRin_T.LWRinTotalGround + (1-ftree)*LWRin_nT.LWRinTotalGround
            LWRin_t.LWRinTotalCanyon = ftree*LWRin_T.LWRinTotalCanyon + (1-ftree)*LWRin_nT.LWRinTotalCanyon

            class LWRout_t_Def():
                pass
            LWRout_t = LWRout_t_Def()
            LWRout_t.LWRoutGroundImp = ftree*LWRout_T.LWRoutGroundImp + (1-ftree)*LWRout_nT.LWRoutGroundImp
            LWRout_t.LWRoutGroundBare = ftree*LWRout_T.LWRoutGroundBare + (1-ftree)*LWRout_nT.LWRoutGroundBare
            LWRout_t.LWRoutGroundVeg = ftree*LWRout_T.LWRoutGroundVeg + (1-ftree)*LWRout_nT.LWRoutGroundVeg
            LWRout_t.LWRoutTree = ftree*LWRout_T.LWRoutTree + (1-ftree)*LWRout_nT.LWRoutTree
            LWRout_t.LWRoutWallSun = ftree*LWRout_T.LWRoutWallSun + (1-ftree)*LWRout_nT.LWRoutWallSun
            LWRout_t.LWRoutWallShade = ftree*LWRout_T.LWRoutWallShade + (1-ftree)*LWRout_nT.LWRoutWallShade
            LWRout_t.LWRoutTotalGround = ftree*LWRout_T.LWRoutTotalGround + (1-ftree)*LWRout_nT.LWRoutTotalGround
            LWRout_t.LWRoutTotalCanyon = ftree*LWRout_T.LWRoutTotalCanyon + (1-ftree)*LWRout_nT.LWRoutTotalCanyon

            class LWRabs_t_Def():
                pass
            LWRabs_t = LWRabs_t_Def()
            LWRabs_t.LWRabsGroundImp = ftree*LWRabs_T.LWRabsGroundImp + (1-ftree)*LWRabs_nT.LWRabsGroundImp
            LWRabs_t.LWRabsGroundBare = ftree*LWRabs_T.LWRabsGroundBare + (1-ftree)*LWRabs_nT.LWRabsGroundBare
            LWRabs_t.LWRabsGroundVeg = ftree*LWRabs_T.LWRabsGroundVeg + (1-ftree)*LWRabs_nT.LWRabsGroundVeg
            LWRabs_t.LWRabsTree = ftree*LWRabs_T.LWRabsTree + (1-ftree)*LWRabs_nT.LWRabsTree
            LWRabs_t.LWRabsWallSun = ftree*LWRabs_T.LWRabsWallSun + (1-ftree)*LWRabs_nT.LWRabsWallSun
            LWRabs_t.LWRabsWallShade = ftree*LWRabs_T.LWRabsWallShade + (1-ftree)*LWRabs_nT.LWRabsWallShade
            LWRabs_t.LWRabsTotalGround = ftree*LWRabs_T.LWRabsTotalGround + (1-ftree)*LWRabs_nT.LWRabsTotalGround
            LWRabs_t.LWRabsTotalCanyon = ftree*LWRabs_T.LWRabsTotalCanyon + (1-ftree)*LWRabs_nT.LWRabsTotalCanyon

            class LWREB_t_Def():
                pass
            LWREB_t = LWREB_t_Def()
            LWREB_t.LWREBGroundImp = ftree*LWREB_T.LWREBGroundImp + (1-ftree)*LWREB_nT.LWREBGroundImp
            LWREB_t.LWREBGroundBare = ftree*LWREB_T.LWREBGroundBare + (1-ftree)*LWREB_nT.LWREBGroundBare
            LWREB_t.LWREBGroundVeg = ftree*LWREB_T.LWREBGroundVeg + (1-ftree)*LWREB_nT.LWREBGroundVeg
            LWREB_t.LWREBTree = ftree*LWREB_T.LWREBTree + (1-ftree)*LWREB_nT.LWREBTree
            LWREB_t.LWREBWallSun = ftree*LWREB_T.LWREBWallSun + (1-ftree)*LWREB_nT.LWREBWallSun
            LWREB_t.LWREBWallShade = ftree*LWREB_T.LWREBWallShade + (1-ftree)*LWREB_nT.LWREBWallShade
            LWREB_t.LWREBTotalGround = ftree*LWREB_T.LWREBTotalGround + (1-ftree)*LWREB_nT.LWREBTotalGround
            LWREB_t.LWREBTotalCanyon = ftree*LWREB_T.LWREBTotalCanyon + (1-ftree)*LWREB_nT.LWREBTotalCanyon

            # The absorbed radiation by the tree is not averaged as it is per tree surface
            LWRin_t.LWRinTree = LWRin_T.LWRinTree
            LWRout_t.LWRoutTree = LWRout_T.LWRoutTree
            LWRabs_t.LWRabsTree = LWRabs_T.LWRabsTree
            LWREB_t.LWREBTree = LWREB_T.LWREBTree

        elif trees == 0:
            # Longwave radiation absorbed without trees
            LWRin_nT, LWRout_nT, LWRabs_nT, LWREB_nT = self.LWRabsorbedNoTree(h_can, w_can, LWR, fgveg,
                                                                                             fgbare, fgimp, ew, egveg,
                                                                                             egbare, egimp, Tgrimp, Tgbare,
                                                                                             Tgveg, Twsun, Twshade, ViewFactor)
            LWRin_t = LWRin_nT
            LWRout_t = LWRout_nT
            LWRabs_t = LWRabs_nT
            LWREB_t = LWREB_nT

        return LWRin_t,LWRout_t,LWRabs_t,LWREB_t

    def TotalSWRabsorbed(self,geometry,FractionsGround,ParTree,PropOpticalGround,PropOpticalWall,PropOpticalTree,
                         ParVegTree,MeteoData,SunPosition,ViewFactor):

        h_can = geometry.hcanyon
        w_can = geometry.wcanyon
        h_tree = geometry.htree
        r_tree = geometry.radius_tree
        d_tree = geometry.distance_tree
        fgveg = FractionsGround.fveg
        fgbare = FractionsGround.fbare
        fgimp = FractionsGround.fimp
        aw = PropOpticalWall.albedo
        agveg = PropOpticalGround.aveg
        agbare = PropOpticalGround.abare
        agimp = PropOpticalGround.aimp
        at = PropOpticalTree.albedo
        trees = ParTree.trees
        LAIt = ParVegTree.LAI
        SWR_dir = MeteoData.SW_dir
        SWR_diff = MeteoData.SW_diff
        theta_Z = SunPosition.theta_Z
        theta_n = SunPosition.theta_n
        ftree = ParTree.ftree

        if trees == 1:
            # Shortwave radiation absorbed without trees
            SWRin_nT, SWRout_nT, SWRabs_nT, SWRabsDir_nT, SWRabsDiff_nT, SWREB_nT = \
                self.SWRabsorbedNoTrees(h_can, w_can, fgveg, fgbare, fgimp, aw, agveg, agbare, agimp, SWR_dir,
                                                   SWR_diff, theta_Z, theta_n, ViewFactor, ParVegTree)

            # Shortwave radiation absorbed with trees
            SWRin_T, SWRout_T, SWRabs_T, SWRabsDir_T, SWRabsDiff_T, SWREB_T = \
                self.SWRabsorbedWithTrees(h_can, w_can, h_tree, r_tree, d_tree, fgveg, fgbare, fgimp, aw, agveg, agbare,
                                      agimp, at, LAIt, SWR_dir, SWR_diff, theta_Z, theta_n, ViewFactor, ParVegTree)

            class SWRin_t_Def():
                pass
            SWRin_t = SWRin_t_Def()
            SWRin_t.SWRinGroundImp = ftree * SWRin_T.SWRinGroundImp + (1 - ftree) * SWRin_nT.SWRinGroundImp
            SWRin_t.SWRinGroundBare = ftree * SWRin_T.SWRinGroundBare + (1 - ftree) * SWRin_nT.SWRinGroundBare
            SWRin_t.SWRinGroundVeg = ftree * SWRin_T.SWRinGroundVeg + (1 - ftree) * SWRin_nT.SWRinGroundVeg
            SWRin_t.SWRinTree = ftree * SWRin_T.SWRinTree + (1 - ftree) * SWRin_nT.SWRinTree
            SWRin_t.SWRinWallSun = ftree * SWRin_T.SWRinWallSun + (1 - ftree) * SWRin_nT.SWRinWallSun
            SWRin_t.SWRinWallShade = ftree * SWRin_T.SWRinWallShade + (1 - ftree) * SWRin_nT.SWRinWallShade
            SWRin_t.SWRinTotalGround = ftree * SWRin_T.SWRinTotalGround + (1 - ftree) * SWRin_nT.SWRinTotalGround
            SWRin_t.SWRinTotalCanyon = ftree * SWRin_T.SWRinTotalCanyon + (1 - ftree) * SWRin_nT.SWRinTotalCanyon

            class SWRout_t_Def():
                pass
            SWRout_t = SWRout_t_Def()
            SWRout_t.SWRoutGroundImp = ftree * SWRout_T.SWRoutGroundImp + (1 - ftree) * SWRout_nT.SWRoutGroundImp
            SWRout_t.SWRoutGroundBare = ftree * SWRout_T.SWRoutGroundBare + (1 - ftree) * SWRout_nT.SWRoutGroundBare
            SWRout_t.SWRoutGroundVeg = ftree * SWRout_T.SWRoutGroundVeg + (1 - ftree) * SWRout_nT.SWRoutGroundVeg
            SWRout_t.SWRoutTree = ftree * SWRout_T.SWRoutTree + (1 - ftree) * SWRout_nT.SWRoutTree
            SWRout_t.SWRoutWallSun = ftree * SWRout_T.SWRoutWallSun + (1 - ftree) * SWRout_nT.SWRoutWallSun
            SWRout_t.SWRoutWallShade = ftree * SWRout_T.SWRoutWallShade + (1 - ftree) * SWRout_nT.SWRoutWallShade
            SWRout_t.SWRoutTotalGround = ftree * SWRout_T.SWRoutTotalGround + (1 - ftree) * SWRout_nT.SWRoutTotalGround
            SWRout_t.SWRoutTotalCanyon = ftree * SWRout_T.SWRoutTotalCanyon + (1 - ftree) * SWRout_nT.SWRoutTotalCanyon

            class SWRabs_t_Def():
                pass

            SWRabs_t = SWRabs_t_Def()
            SWRabs_t.SWRabsGroundImp = ftree * SWRabs_T.SWRabsGroundImp + (1 - ftree) * SWRabs_nT.SWRabsGroundImp
            SWRabs_t.SWRabsGroundBare = ftree * SWRabs_T.SWRabsGroundBare + (1 - ftree) * SWRabs_nT.SWRabsGroundBare
            SWRabs_t.SWRabsGroundVeg = ftree * SWRabs_T.SWRabsGroundVeg + (1 - ftree) * SWRabs_nT.SWRabsGroundVeg
            SWRabs_t.SWRabsTree = ftree * SWRabs_T.SWRabsTree + (1 - ftree) * SWRabs_nT.SWRabsTree
            SWRabs_t.SWRabsWallSun = ftree * SWRabs_T.SWRabsWallSun + (1 - ftree) * SWRabs_nT.SWRabsWallSun
            SWRabs_t.SWRabsWallShade = ftree * SWRabs_T.SWRabsWallShade + (1 - ftree) * SWRabs_nT.SWRabsWallShade
            SWRabs_t.SWRabsTotalGround = ftree * SWRabs_T.SWRabsTotalGround + (1 - ftree) * SWRabs_nT.SWRabsTotalGround
            SWRabs_t.SWRabsTotalCanyon = ftree * SWRabs_T.SWRabsTotalCanyon + (1 - ftree) * SWRabs_nT.SWRabsTotalCanyon

            class SWRabsDir_t_Def():
                pass

            SWRabsDir_t = SWRabsDir_t_Def()
            SWRabsDir_t.SWRabsGroundImp = ftree * SWRabsDir_T.SWRabsGroundImp + (1 - ftree) * SWRabsDir_nT.SWRabsGroundImp
            SWRabsDir_t.SWRabsGroundBare = ftree * SWRabsDir_T.SWRabsGroundBare + (1 - ftree) * SWRabsDir_nT.SWRabsGroundBare
            SWRabsDir_t.SWRabsGroundVeg = ftree * SWRabsDir_T.SWRabsGroundVeg + (1 - ftree) * SWRabsDir_nT.SWRabsGroundVeg
            SWRabsDir_t.SWRabsTree = ftree * SWRabsDir_T.SWRabsTree + (1 - ftree) * SWRabsDir_nT.SWRabsTree
            SWRabsDir_t.SWRabsWallSun = ftree * SWRabsDir_T.SWRabsWallSun + (1 - ftree) * SWRabsDir_nT.SWRabsWallSun
            SWRabsDir_t.SWRabsWallShade = ftree * SWRabsDir_T.SWRabsWallShade + (1 - ftree) * SWRabsDir_nT.SWRabsWallShade
            SWRabsDir_t.SWRabsTotalGround = ftree * SWRabsDir_T.SWRabsTotalGround + (1 - ftree) * SWRabsDir_nT.SWRabsTotalGround
            SWRabsDir_t.SWRabsTotalCanyon = ftree * SWRabs_T.SWRabsTotalCanyon + (1 - ftree) * SWRabsDir_nT.SWRabsTotalCanyon

            class SWRabsDiff_t_Def():
                pass

            SWRabsDiff_t = SWRabsDiff_t_Def()
            SWRabsDiff_t.SWRabsGroundImp = ftree * SWRabsDiff_T.SWRabsGroundImp + (1 - ftree) * SWRabsDiff_nT.SWRabsGroundImp
            SWRabsDiff_t.SWRabsGroundBare = ftree * SWRabsDiff_T.SWRabsGroundBare + (1 - ftree) * SWRabsDiff_nT.SWRabsGroundBare
            SWRabsDiff_t.SWRabsGroundVeg = ftree * SWRabsDiff_T.SWRabsGroundVeg + (1 - ftree) * SWRabsDiff_nT.SWRabsGroundVeg
            SWRabsDiff_t.SWRabsTree = ftree * SWRabsDiff_T.SWRabsTree + (1 - ftree) * SWRabsDiff_nT.SWRabsTree
            SWRabsDiff_t.SWRabsWallSun = ftree * SWRabsDiff_T.SWRabsWallSun + (1 - ftree) * SWRabsDiff_nT.SWRabsWallSun
            SWRabsDiff_t.SWRabsWallShade = ftree * SWRabsDiff_T.SWRabsWallShade + (1 - ftree) * SWRabsDiff_nT.SWRabsWallShade
            SWRabsDiff_t.SWRabsTotalGround = ftree * SWRabsDiff_T.SWRabsTotalGround + (1 - ftree) * SWRabsDiff_nT.SWRabsTotalGround
            SWRabsDiff_t.SWRabsTotalCanyon = ftree * SWRabs_T.SWRabsTotalCanyon + (1 - ftree) * SWRabsDiff_nT.SWRabsTotalCanyon

            class SWREB_t_Def():
                pass

            SWREB_t = SWREB_t_Def()
            SWREB_t.SWREBGroundImp = ftree * SWREB_T.SWREBGroundImp + (1 - ftree) * SWREB_nT.SWREBGroundImp
            SWREB_t.SWREBGroundBare = ftree * SWREB_T.SWREBGroundBare + (1 - ftree) * SWREB_nT.SWREBGroundBare
            SWREB_t.SWREBGroundVeg = ftree * SWREB_T.SWREBGroundVeg + (1 - ftree) * SWREB_nT.SWREBGroundVeg
            SWREB_t.SWREBTree = ftree * SWREB_T.SWREBTree + (1 - ftree) * SWREB_nT.SWREBTree
            SWREB_t.SWREBWallSun = ftree * SWREB_T.SWREBWallSun + (1 - ftree) * SWREB_nT.SWREBWallSun
            SWREB_t.SWREBWallShade = ftree * SWREB_T.SWREBWallShade + (1 - ftree) * SWREB_nT.SWREBWallShade
            SWREB_t.SWREBTotalGround = ftree * SWREB_T.SWREBTotalGround + (1 - ftree) * SWREB_nT.SWREBTotalGround
            SWREB_t.SWREBTotalCanyon = ftree * SWREB_T.SWREBTotalCanyon + (1 - ftree) * SWREB_nT.SWREBTotalCanyon

            # The absorbed radiation by the tree is not averaged as it is per tree surface
            SWRin_t.SWRinTree = SWRin_T.SWRinTree
            SWRout_t.SWRoutTree = SWRout_T.SWRoutTree
            SWRabs_t.SWRabsTree = SWRabs_T.SWRabsTree
            SWRabsDir_t.SWRabsTree = SWRabsDir_T.SWRabsTree
            SWRabsDiff_t.SWRabsTree = SWRabsDiff_T.SWRabsTree
            SWREB_t.SWREBTree = SWREB_T.SWREBTree

        elif trees == 0:
            SWRin_nT, SWRout_nT, SWRabs_nT, SWRabsDir_nT, SWRabsDiff_nT, SWREB_nT = \
                self.SWRabsorbedNoTrees(h_can, w_can, fgveg, fgbare, fgimp, aw, agveg, agbare, agimp, SWR_dir, SWR_diff,
                                    theta_Z, theta_n, ViewFactor, ParVegTree)

            SWRin_t = SWRin_nT
            SWRout_t = SWRout_nT
            SWRabs_t = SWRabs_nT
            SWRabsDir_t = SWRabsDir_nT
            SWRabsDiff_t = SWRabsDiff_nT
            SWREB_t = SWREB_nT
        return SWRin_t,SWRout_t,SWRabs_t,SWRabsDir_t,SWRabsDiff_t,SWREB_t


    def LWRabsorbedNoTree(self,h_can,w_can,LWR,fgveg,fgbare,fgimp,ew,egveg,egbare,egimp,Tgimp,Tgbare,Tgveg,Twsun,Twshade,ViewFactor):

        F_gs_nT = ViewFactor.F_gs_nT
        F_gw_nT = ViewFactor.F_gw_nT
        F_ww_nT = ViewFactor.F_ww_nT
        F_wg_nT = ViewFactor.F_wg_nT
        F_ws_nT = ViewFactor.F_ws_nT
        F_sg_nT = ViewFactor.F_sg_nT
        F_sw_nT = ViewFactor.F_sw_nT

        # normalized surface areas
        A_s = w_can
        A_g = w_can
        A_w = h_can
        bolzm = 5.67 * 10 ** (-8)

        SVF = numpy.zeros(3)
        SVF[0] = F_gs_nT + 2 * F_gw_nT
        SVF[1] = F_ww_nT + F_wg_nT + F_ws_nT
        SVF[2] = F_sg_nT + 2 * F_sw_nT

        SVF2 = numpy.zeros(3)
        SVF2[0] = F_gs_nT + 2 * F_ws_nT * h_can
        SVF2[1] = F_sg_nT + 2 * F_wg_nT * h_can
        SVF2[2] = F_ww_nT + F_sw_nT / h_can + F_gw_nT / h_can

        # Solve for infinite reflections equation A*X=C
        if fgimp > 0:
            Cimp = 1
        else:
            Cimp = 0
        if fgbare > 0:
            Cbare = 1
        else:
            Cbare = 0
        if fgveg > 0:
            Cveg = 1
        else:
            Cveg = 0

        #  View factor matrix to solve for infinite reflections equation
        Tij = numpy.array([[1,0,0, -(1-egveg)*F_gw_nT*Cveg, -(1-egveg)*F_gw_nT*Cveg, -(1-egveg)*F_gs_nT*Cveg],
               [0,1,0, -(1-egbare)*F_gw_nT*Cbare, -(1-egbare)*F_gw_nT*Cbare, -(1-egbare)*F_gs_nT*Cbare],
               [0,0,1, -(1-egimp)*F_gw_nT*Cimp, -(1-egimp)*F_gw_nT*Cimp, -(1-egimp)*F_gs_nT*Cimp],
               [-(1-ew)*F_wg_nT*fgveg*Cveg, -(1-ew)*F_wg_nT*fgbare*Cbare, -(1-ew)*F_wg_nT*fgimp*Cimp, 1, -(1-ew)*F_ww_nT, -(1-ew)*F_ws_nT],
               [-(1-ew)*F_wg_nT*fgveg*Cveg, -(1-ew)*F_wg_nT*fgbare*Cbare, -(1-ew)*F_wg_nT*fgimp*Cimp, -(1-ew)*F_ww_nT, 1, -(1-ew)*F_ws_nT],
               [0, 0, 0, 0, 0, 1]])

        # Emitted radiation per surface
        Omega_i = numpy.array([(egveg*bolzm*(Tgveg)**4*Cveg),
                               (egbare * bolzm * (Tgbare) ** 4 * Cbare),
                               (egimp * bolzm * (Tgimp) ** 4 * Cimp),
                               (ew * bolzm * (Twsun) ** 4),
                               (ew * bolzm * (Twshade) ** 4),
                               LWR])

        # Outgoing radiation per surface
        # Outgoing radiation [W m^-2]

        B_i = numpy.linalg.solve(Tij,Omega_i)

        # Incoming longwave radiation at each surface A_i
        Tij2 = numpy.array([[0, 0, 0, F_gw_nT*Cveg, F_gw_nT*Cveg, F_gs_nT*Cveg],
                [0, 0, 0, F_gw_nT*Cbare, F_gw_nT*Cbare, F_gs_nT*Cbare],
                [0, 0, 0, F_gw_nT*Cimp, F_gw_nT*Cimp, F_gs_nT*Cimp],
                [F_wg_nT*fgveg*Cveg, F_wg_nT*fgbare*Cbare, F_wg_nT*fgimp*Cimp, 0, F_ww_nT, F_ws_nT],
                [F_wg_nT*fgveg*Cveg, F_wg_nT*fgbare*Cbare, F_wg_nT*fgimp*Cimp, F_ww_nT, 0, F_ws_nT],
                [0, 0, 0, 0, 0, 0]])

        A_i = numpy.dot(Tij2,B_i)

        # Absorbed longwave radiation
        e_i = [egveg, egbare, egimp, ew, ew, 0]
        Qnet_i = [(e_i[i] * B_i[i] - Omega_i[i]) / (1 - e_i[i]) for i in range(0,len(e_i))]
        for i in range(0,len(e_i)):
            if e_i[i] == 1:
                Qnet_i[i] = A_i[i] - Omega_i[i]

        # Assumption: The sky has a fixed emission of LWR. Hence, Qnet is 0
        Qnet_i[5] = 0

        # Assignment
        # Outgoing radiation [W m^-2]
        LWRout_i = B_i
        # Emitted radiation [W m^-2]
        LWRemit_i = Omega_i
        # Incoming radiation [W m^-2]
        LWRin_i = A_i
        # Net absorbed radiation [W m^-2]
        LWRnet_i = Qnet_i

        # Energy Balance
        LWRin_atm = LWR
        TotalLWRSurface_in = LWRin_i[0]*fgveg*A_g/A_g + LWRin_i[1]*fgbare*A_g/A_g + LWRin_i[2]*fgimp*A_g/A_g +\
                             LWRin_i[3]*A_w/A_g + LWRin_i[4]*A_w/A_g

        TotalLWRSurface_abs	=	LWRnet_i[0]*fgveg*A_g/A_g + LWRnet_i[1]*fgbare*A_g/A_g + LWRnet_i[2]*fgimp*A_g/A_g +\
                                LWRnet_i[3]*A_w/A_g + LWRnet_i[4]*A_w/A_g

        TotalLWRSurface_out	=	LWRout_i[0]*fgveg*A_g/A_s+LWRout_i[1]*fgbare*A_g/A_s+LWRout_i[2]*fgimp*A_g/A_s +\
                                LWRout_i[3]*A_w/A_s+LWRout_i[4]*A_w/A_s



        TotalLWRref_to_atm	=	LWRout_i[0]*F_sg_nT*fgveg + LWRout_i[1]*F_sg_nT*fgbare + LWRout_i[2]*F_sg_nT*fgimp + \
                                LWRout_i[3]*F_sw_nT + LWRout_i[4]*F_sw_nT

        class LWRin_nT_Def():
            pass
        LWRin_nT = LWRin_nT_Def()
        LWRin_nT.LWRinGroundImp = LWRin_i[2] * Cimp
        LWRin_nT.LWRinGroundBare = LWRin_i[1] * Cbare
        LWRin_nT.LWRinGroundVeg = LWRin_i[0] * Cveg
        LWRin_nT.LWRinTree = 0
        LWRin_nT.LWRinWallSun = LWRin_i[3]
        LWRin_nT.LWRinWallShade = LWRin_i[4]
        LWRin_nT.LWRinTotalGround = fgveg * LWRin_i[0] + fgbare * LWRin_i[1] + fgimp * LWRin_i[2]
        LWRin_nT.LWRinTotalCanyon = LWRin_i[0] * fgveg * A_g / A_g + LWRin_i[1] * fgbare * A_g / A_g + \
                                    LWRin_i[2] * fgimp * A_g / A_g + LWRin_i[3] * A_w / A_g + LWRin_i[4] * A_w / A_g

        # Outgoing longwave radiation
        class LWRout_nT_Def():
            pass
        LWRout_nT = LWRout_nT_Def()
        LWRout_nT.LWRoutGroundImp = LWRout_i[2] * Cimp
        LWRout_nT.LWRoutGroundBare = LWRout_i[1] * Cbare
        LWRout_nT.LWRoutGroundVeg = LWRout_i[0] * Cveg
        LWRout_nT.LWRoutTree = 0
        LWRout_nT.LWRoutWallSun = LWRout_i[3]
        LWRout_nT.LWRoutWallShade = LWRout_i[4]
        LWRout_nT.LWRoutTotalGround = fgveg * LWRout_i[0] + fgbare * LWRout_i[1] + fgimp * LWRout_i[2]
        LWRout_nT.LWRoutTotalCanyon = LWRout_i[0] * fgveg * A_g / A_g + LWRout_i[1] * fgbare * A_g / A_g + \
                                      LWRout_i[2] * fgimp * A_g / A_g + LWRout_i[3] * A_w / A_g + LWRout_i[4] * A_w / A_g

        # Absorbed longwave radiation
        class LWRabs_nT_Def():
            pass
        LWRabs_nT = LWRabs_nT_Def()
        LWRabs_nT.LWRabsGroundImp = LWRnet_i[2] * Cimp
        LWRabs_nT.LWRabsGroundBare = LWRnet_i[1] * Cbare
        LWRabs_nT.LWRabsGroundVeg = LWRnet_i[0] * Cveg
        LWRabs_nT.LWRabsTree = 0
        LWRabs_nT.LWRabsWallSun = LWRnet_i[3]
        LWRabs_nT.LWRabsWallShade = LWRnet_i[4]
        LWRabs_nT.LWRabsTotalGround = fgveg * LWRnet_i[0] + fgbare * LWRnet_i[1] + fgimp * LWRnet_i[2]
        LWRabs_nT.LWRabsTotalCanyon = LWRnet_i[0] * fgveg * A_g / A_g + LWRnet_i[1] * fgbare * A_g / A_g + \
                                      LWRnet_i[2] * fgimp * A_g / A_g + LWRnet_i[3] * A_w / A_g + LWRnet_i[4] * A_w / A_g

        # Energy Balance of longwave radiation
        class LWREB_nT_Def():
            pass
        LWREB_nT = LWREB_nT_Def()
        LWREB_nT.LWREBGroundImp = LWRin_nT.LWRinGroundImp - LWRout_nT.LWRoutGroundImp - LWRabs_nT.LWRabsGroundImp
        LWREB_nT.LWREBGroundBare = LWRin_nT.LWRinGroundBare - LWRout_nT.LWRoutGroundBare - LWRabs_nT.LWRabsGroundBare
        LWREB_nT.LWREBGroundVeg = LWRin_nT.LWRinGroundVeg - LWRout_nT.LWRoutGroundVeg - LWRabs_nT.LWRabsGroundVeg
        LWREB_nT.LWREBTree = 0
        LWREB_nT.LWREBWallSun = LWRin_nT.LWRinWallSun - LWRout_nT.LWRoutWallSun - LWRabs_nT.LWRabsWallSun
        LWREB_nT.LWREBWallShade = LWRin_nT.LWRinWallShade - LWRout_nT.LWRoutWallShade - LWRabs_nT.LWRabsWallShade
        LWREB_nT.LWREBTotalGround = LWRin_nT.LWRinTotalGround - LWRout_nT.LWRoutTotalGround - LWRabs_nT.LWRabsTotalGround
        LWREB_nT.LWREBTotalCanyon = LWRin_nT.LWRinTotalCanyon - LWRout_nT.LWRoutTotalCanyon - LWRabs_nT.LWRabsTotalCanyon

        if abs(LWREB_nT.LWREBGroundImp)>=10**(-6):
            print('LWREB_nT.LWREBGroundImp is not 0.')
        if abs(LWREB_nT.LWREBGroundBare)>=10**(-6):
            print('LWREB_nT.LWREBGroundBare is not 0.')
        if abs(LWREB_nT.LWREBGroundVeg	)>=10**(-6):
            print('LWREB_nT.LWREBGroundVeg	 is not 0.')
        if abs(LWREB_nT.LWREBWallSun)>=10**(-6):
            print('LWREB_nT.LWREBWallSun is not 0.')
        if abs(LWREB_nT.LWREBWallShade)>=10**(-6):
            print('LWREB_nT.LWREBWallShade is not 0.')
        if abs(LWREB_nT.LWREBTotalGround)>=10**(-6):
            print('LWREB_nT.LWREBTotalGround is not 0.')
        if abs(LWREB_nT.LWREBTotalCanyon)>=10**(-6):
            print('LWREB_nT.LWREBTotalCanyon is not 0.')

        return LWRin_nT,LWRout_nT,LWRabs_nT,LWREB_nT

    def LWRabsorbedWithTrees(self,h_can,w_can,r_tree,LWR,fgveg,fgbare,fgimp,ew,et,egveg,egbare,egimp,Tgimp,Tgbare,Tgveg,
                             Twsun,Twshade,Ttree,ViewFactor):
        F_gs_T = ViewFactor.F_gs_T
        F_gt_T = ViewFactor.F_gt_T
        F_gw_T = ViewFactor.F_gw_T
        F_ww_T = ViewFactor.F_ww_T
        F_wt_T = ViewFactor.F_wt_T
        F_wg_T = ViewFactor.F_wg_T
        F_ws_T = ViewFactor.F_ws_T
        F_sg_T = ViewFactor.F_sg_T
        F_sw_T = ViewFactor.F_sw_T
        F_st_T = ViewFactor.F_st_T
        F_tg_T = ViewFactor.F_tg_T
        F_tw_T = ViewFactor.F_tw_T
        F_ts_T = ViewFactor.F_ts_T
        F_tt_T = ViewFactor.F_tt_T

        # normalized surface areas
        A_s = w_can
        A_g = w_can
        A_w = h_can
        # There are 2 trees. Hence, the area of tree is twice a circle
        A_t = 2 * 2 * numpy.pi * r_tree
        bolzm = 5.67 * 10 ** (-8)

        # Check if view factors add up to 1
        SVF = numpy.zeros(4)
        SVF[0] = F_gs_T + F_gt_T + 2 * F_gw_T
        SVF[1] = F_ww_T + F_wt_T + F_wg_T + F_ws_T
        SVF[2] = F_sg_T + 2 * F_sw_T + F_st_T
        SVF[3] = F_ts_T + 2 * F_tw_T + F_tt_T + F_tg_T

        SVF2 = numpy.zeros(4)
        SVF2[0] = F_gs_T + 2 * F_ws_T + F_ts_T
        SVF2[1] = F_sg_T + 2 * F_wg_T + F_tg_T
        SVF2[2] = F_ww_T + F_sw_T + F_gw_T + F_tw_T
        SVF2[3] = F_gt_T + 2 * F_wt_T + F_tt_T + F_st_T

        # Solve for infinite reflections equation A*X=C
        if fgimp > 0:
            Cimp = 1
        else:
            Cimp = 0
        if fgbare > 0:
            Cbare = 1
        else:
            Cbare = 0
        if fgveg > 0:
            Cveg = 1
        else:
            Cveg = 0

        Tij = numpy.array([[1,0,0, -(1-egveg)*F_gw_T*Cveg, -(1-egveg)*F_gw_T*Cveg, -(1-egveg)*F_gt_T*Cveg, -(1-egveg)*F_gs_T*Cveg],
               [0,1,0, -(1-egbare)*F_gw_T*Cbare, -(1-egbare)*F_gw_T*Cbare, -(1-egbare)*F_gt_T*Cbare, -(1-egbare)*F_gs_T*Cbare],
               [0,0,1, -(1-egimp)*F_gw_T*Cimp, -(1-egimp)*F_gw_T*Cimp, -(1-egimp)*F_gt_T*Cimp, -(1-egimp)*F_gs_T*Cimp],
               [-(1-ew)*F_wg_T*fgveg*Cveg, -(1-ew)*F_wg_T*fgbare*Cbare, -(1-ew)*F_wg_T*fgimp*Cimp, 1, -(1-ew)*F_ww_T, -(1-ew)*F_wt_T, -(1-ew)*F_ws_T],
               [-(1-ew)*F_wg_T*fgveg*Cveg, -(1-ew)*F_wg_T*fgbare*Cbare, -(1-ew)*F_wg_T*fgimp*Cimp, -(1-ew)*F_ww_T, 1, -(1-ew)*F_wt_T, -(1-ew)*F_ws_T],
               [-(1-et)*F_tg_T*fgveg*Cveg, -(1-et)*F_tg_T*fgbare*Cbare, -(1-et)*F_tg_T*fgimp*Cimp, -(1-et)*F_tw_T, -(1-et)*F_tw_T, 1-(1-et)*F_tt_T, -(1-et)*F_ts_T],
               [0, 0, 0, 0, 0, 0, 1]])

        Omega_i = numpy.array([(egveg*bolzm*(Tgveg)**4*Cveg),
                   (egbare * bolzm * (Tgbare) ** 4 * Cbare),
                   (egimp * bolzm * (Tgimp) ** 4 * Cimp),
                   (ew * bolzm * (Twsun) ** 4),
                   (ew * bolzm * (Twshade) ** 4),
                   (et * bolzm * (Ttree) ** 4),
                   LWR])

        # Outgoing radiation per surface
        # Outgoing radiation [W m^-2]
        B_i = numpy.linalg.solve(Tij,Omega_i)

        Tij2 = numpy.array([[0, 0, 0, F_gw_T*Cveg, F_gw_T*Cveg, F_gt_T*Cveg, F_gs_T*Cveg],
                [0, 0, 0, F_gw_T*Cbare, F_gw_T*Cbare, F_gt_T*Cbare, F_gs_T*Cbare],
                [0, 0, 0, F_gw_T*Cimp, F_gw_T*Cimp, F_gt_T*Cimp, F_gs_T*Cimp],
                [F_wg_T*fgveg*Cveg, F_wg_T*fgbare*Cbare, F_wg_T*fgimp*Cimp, 0, F_ww_T, F_wt_T, F_ws_T],
                [F_wg_T*fgveg*Cveg, F_wg_T*fgbare*Cbare, F_wg_T*fgimp*Cimp, F_ww_T, 0, F_wt_T, F_ws_T],
                [F_tg_T*fgveg*Cveg, F_tg_T*fgbare*Cbare, F_tg_T*fgimp*Cimp, F_tw_T, F_tw_T, F_tt_T, F_ts_T],
                [0, 0, 0, 0, 0, 0, 0]])

        A_i = numpy.dot(Tij2, B_i)
        e_i = [egveg, egbare, egimp, ew, ew, et, 0]

        # Absorbed longwave radiation
        e_i = [egveg, egbare, egimp, ew, ew, et, 0]
        Qnet_i = [(e_i[i] * B_i[i] - Omega_i[i]) / (1 - e_i[i]) for i in range(0, len(e_i))]
        for i in range(0, len(e_i)):
            if e_i[i] == 1:
                Qnet_i[i] = A_i[i] - Omega_i[i]

        # Assumption: The sky has a fixed emission of LWR. Hence, Qnet is 0
        Qnet_i[6] = 0

        # Assignment
        # Outgoing radiation [W m^-2]
        LWRout_i = B_i
        # Incoming radiation [W m^-2]
        LWRin_i = A_i
        # Net absorbed radiation [W m^-2]
        LWRnet_i = Qnet_i

        # Energy balance
        LWRin_atm = LWR
        TotalLWRSurface_in = LWRin_i[0]*fgveg*A_g/A_g + LWRin_i[1]*fgbare*A_g/A_g + LWRin_i[2]*fgimp*A_g/A_g +\
                             LWRin_i[3]*A_w/A_g + LWRin_i[4]*A_w/A_g + LWRin_i[5]*A_t/A_g

        TotalLWRSurface_abs	=	LWRnet_i[0]*fgveg*A_g/A_g + LWRnet_i[1]*fgbare*A_g/A_g + LWRnet_i[2]*fgimp*A_g/A_g +\
                                LWRnet_i[3]*A_w/A_g + LWRnet_i[4]*A_w/A_g + LWRnet_i[5]*A_t/A_g

        TotalLWRSurface_out	=	LWRout_i[0]*fgveg*A_g/A_s+LWRout_i[1]*fgbare*A_g/A_s+LWRout_i[2]*fgimp*A_g/A_s +\
                                LWRout_i[3]*A_w/A_s+LWRout_i[4]*A_w/A_s+LWRout_i[5]*A_t/A_s

        TotalLWRref_to_atm	=	LWRout_i[0]*F_sg_T*fgveg + LWRout_i[1]*F_sg_T*fgbare + LWRout_i[2]*F_sg_T*fgimp + \
                                LWRout_i[3]*F_sw_T + LWRout_i[4]*F_sw_T + LWRout_i[5]*F_st_T

        class LWRin_T_Def():
            pass
        LWRin_T = LWRin_T_Def()
        LWRin_T.LWRinGroundImp = LWRin_i[2] * Cimp
        LWRin_T.LWRinGroundBare = LWRin_i[1] * Cbare
        LWRin_T.LWRinGroundVeg = LWRin_i[0] * Cveg
        LWRin_T.LWRinTree = LWRin_i[5]
        LWRin_T.LWRinWallSun = LWRin_i[3]
        LWRin_T.LWRinWallShade = LWRin_i[4]
        LWRin_T.LWRinTotalGround = fgveg * LWRin_i[0] + fgbare * LWRin_i[1] + fgimp * LWRin_i[2]
        LWRin_T.LWRinTotalCanyon = LWRin_i[0]*fgveg*A_g/A_g + LWRin_i[1]*fgbare*A_g/A_g + \
                                   LWRin_i[2]*fgimp*A_g/A_g + LWRin_i[3]*A_w/A_g + LWRin_i[4]*A_w/A_g + LWRin_i[5]*A_t/A_g

        # Outgoing longwave radiation
        class LWRout_T_Def():
            pass
        LWRout_T = LWRout_T_Def()
        LWRout_T.LWRoutGroundImp = LWRout_i[2] * Cimp
        LWRout_T.LWRoutGroundBare = LWRout_i[1] * Cbare
        LWRout_T.LWRoutGroundVeg = LWRout_i[0] * Cveg
        LWRout_T.LWRoutTree = LWRout_i[5]
        LWRout_T.LWRoutWallSun = LWRout_i[3]
        LWRout_T.LWRoutWallShade = LWRout_i[4]
        LWRout_T.LWRoutTotalGround = fgveg * LWRout_i[0] + fgbare * LWRout_i[1] + fgimp * LWRout_i[2]
        LWRout_T.LWRoutTotalCanyon = LWRout_i[0] * fgveg * A_g / A_g + LWRout_i[1] * fgbare * A_g / A_g + \
                                      LWRout_i[2] * fgimp * A_g / A_g + LWRout_i[3] * A_w / A_g + LWRout_i[4] * A_w / A_g + LWRout_i[5]*A_t/A_g

        # Absorbed longwave radiation
        class LWRabs_T_Def():
            pass
        LWRabs_T = LWRabs_T_Def()
        LWRabs_T.LWRabsGroundImp = LWRnet_i[2] * Cimp
        LWRabs_T.LWRabsGroundBare = LWRnet_i[1] * Cbare
        LWRabs_T.LWRabsGroundVeg = LWRnet_i[0] * Cveg
        LWRabs_T.LWRabsTree = LWRnet_i[5]
        LWRabs_T.LWRabsWallSun = LWRnet_i[3]
        LWRabs_T.LWRabsWallShade = LWRnet_i[4]
        LWRabs_T.LWRabsTotalGround = fgveg * LWRnet_i[0] + fgbare * LWRnet_i[1] + fgimp * LWRnet_i[2]
        LWRabs_T.LWRabsTotalCanyon = LWRnet_i[0] * fgveg * A_g / A_g + LWRnet_i[1] * fgbare * A_g / A_g + \
                                      LWRnet_i[2] * fgimp * A_g / A_g + LWRnet_i[3] * A_w / A_g + LWRnet_i[4] * A_w / A_g + LWRnet_i[5]*A_t/A_g

        # Energy Balance of longwave radiation
        class LWREB_T_Def():
            pass
        LWREB_T = LWREB_T_Def()
        LWREB_T.LWREBGroundImp = LWRin_T.LWRinGroundImp - LWRout_T.LWRoutGroundImp - LWRabs_T.LWRabsGroundImp
        LWREB_T.LWREBGroundBare = LWRin_T.LWRinGroundBare - LWRout_T.LWRoutGroundBare - LWRabs_T.LWRabsGroundBare
        LWREB_T.LWREBGroundVeg = LWRin_T.LWRinGroundVeg - LWRout_T.LWRoutGroundVeg - LWRabs_T.LWRabsGroundVeg
        LWREB_T.LWREBTree = LWRin_T.LWRinTree - LWRout_T.LWRoutTree - LWRabs_T.LWRabsTree
        LWREB_T.LWREBWallSun = LWRin_T.LWRinWallSun - LWRout_T.LWRoutWallSun - LWRabs_T.LWRabsWallSun
        LWREB_T.LWREBWallShade = LWRin_T.LWRinWallShade - LWRout_T.LWRoutWallShade - LWRabs_T.LWRabsWallShade
        LWREB_T.LWREBTotalGround = LWRin_T.LWRinTotalGround - LWRout_T.LWRoutTotalGround - LWRabs_T.LWRabsTotalGround
        LWREB_T.LWREBTotalCanyon = LWRin_T.LWRinTotalCanyon - LWRout_T.LWRoutTotalCanyon - LWRabs_T.LWRabsTotalCanyon

        if abs(LWREB_T.LWREBGroundImp)>=10**(-6):
            print('LWREB_T.LWREBGroundImp is not 0.')
        if abs(LWREB_T.LWREBGroundBare)>=10**(-6):
            print('LWREB_T.LWREBGroundBare is not 0.')
        if abs(LWREB_T.LWREBGroundVeg	)>=10**(-6):
            print('LWREB_T.LWREBGroundVeg	 is not 0.')
        if abs(LWREB_T.LWREBWallSun)>=10**(-6):
            print('LWREB_T.LWREBWallSun is not 0.')
        if abs(LWREB_T.LWREBWallShade)>=10**(-6):
            print('LWREB_T.LWREBWallShade is not 0.')
        if abs(LWREB_T.LWREBTotalGround)>=10**(-6):
            print('LWREB_T.LWREBTotalGround is not 0.')
        if abs(LWREB_T.LWREBTotalCanyon)>=10**(-6):
            print('LWREB_T.LWREBTotalCanyon is not 0.')
        if abs(LWREB_T.LWREBTree)>=10**(-6):
            print('LWREB_T.LWREBTree is not 0.')

        return LWRin_T,LWRout_T,LWRabs_T,LWREB_T


    def SWRabsorbedNoTrees(self,h_can,w_can,fgveg,fgbare,fgimp,aw,agveg,agbare,agimp,SWR_dir,SWR_diff,theta_Z,theta_n,
                           ViewFactor,ParVegTree):
        F_gs_nT = ViewFactor.F_gs_nT
        F_gw_nT = ViewFactor.F_gw_nT
        F_ww_nT = ViewFactor.F_ww_nT
        F_wg_nT = ViewFactor.F_wg_nT
        F_ws_nT = ViewFactor.F_ws_nT
        F_sg_nT = ViewFactor.F_sg_nT
        F_sw_nT = ViewFactor.F_sw_nT

        # normalized surface areas
        A_s = w_can
        A_g = w_can
        A_w = h_can

        SVF = numpy.zeros(3)
        SVF[0] = F_gs_nT + 2 * F_gw_nT
        SVF[1] = F_ww_nT + F_wg_nT + F_ws_nT
        SVF[2] = F_sg_nT + 2 * F_sw_nT

        SVF2 = numpy.zeros(3)
        SVF2[0] = F_gs_nT + 2 * F_ws_nT * h_can
        SVF2[1] = F_sg_nT + 2 * F_wg_nT * h_can
        SVF2[2] = F_ww_nT + F_sw_nT / h_can + F_gw_nT / h_can

        SWRdir_ground, SWRdir_wallsun, SWRdir_wallshade, NOTUSED = self.DirectSWRSurfaces(h_can, w_can, numpy.nan, numpy.nan, numpy.nan,
                                                                                           theta_Z,theta_n, SWR_dir, numpy.nan,
                                                                                          0,ParVegTree)

        if fgimp > 0:
            Cimp = 1
        else:
            Cimp = 0
        if fgbare > 0:
            Cbare = 1
        else:
            Cbare = 0
        if fgveg > 0:
            Cveg = 1
        else:
            Cveg = 0

        ai = [agveg,agbare,agimp,aw,aw,0]

        # View factor matrix to solve for infinite reflections equation
        Tij = numpy.array([[1,0,0, -agveg*F_gw_nT*Cveg, -agveg*F_gw_nT*Cveg, -agveg*F_gs_nT*Cveg],
               [0,1,0, -agbare*F_gw_nT*Cbare, -agbare*F_gw_nT*Cbare, -agbare*F_gs_nT*Cbare],
               [0,0,1, -agimp*F_gw_nT*Cimp, -agimp*F_gw_nT*Cimp, -agimp*F_gs_nT*Cimp],
               [-aw*F_wg_nT*fgveg*Cveg,-aw*F_wg_nT*fgbare*Cbare,-aw*F_wg_nT*fgimp*Cimp, 1, -aw*F_ww_nT, -aw*F_ws_nT],
               [-aw*F_wg_nT*fgveg*Cveg,-aw*F_wg_nT*fgbare*Cbare,-aw*F_wg_nT*fgimp*Cimp, -aw*F_ww_nT, 1, -aw*F_ws_nT],
               [0, 0, 0, 0, 0, 1]])

        # Incoming shortwave radiation from sky
        Omega_i = numpy.array([agveg*SWRdir_ground*Cveg,
                   agbare * SWRdir_ground * Cbare,
                   agimp * SWRdir_ground * Cimp,
                   aw * SWRdir_wallsun,
                   aw * 0,
                   SWR_diff])

        # Outgoing radiation per surface
        B_i = numpy.linalg.solve(Tij,Omega_i)

        # Incoming shortwave radiation at each surface A_i
        Tij2 = numpy.array([[0, 0, 0, F_gw_nT*Cveg, F_gw_nT*Cveg, F_gs_nT*Cveg],
                [0, 0, 0, F_gw_nT*Cbare, F_gw_nT*Cbare, F_gs_nT*Cbare],
                [0, 0, 0, F_gw_nT*Cimp, F_gw_nT*Cimp, F_gs_nT*Cimp],
                [F_wg_nT*fgveg*Cveg, F_wg_nT*fgbare*Cbare, F_wg_nT*fgimp*Cimp, 0, F_ww_nT, F_ws_nT],
                [F_wg_nT*fgveg*Cveg, F_wg_nT*fgbare*Cbare, F_wg_nT*fgimp*Cimp, F_ww_nT, 0, F_ws_nT],
                [0, 0, 0, 0, 0, 0]])

        SWRdir_i = numpy.array([SWRdir_ground*Cveg,
                    SWRdir_ground * Cbare,
                    SWRdir_ground * Cimp,
                    SWRdir_wallsun,
                    0,
                    0])

        # Incoming radiation [W m^-2]
        A_i1	=	numpy.dot(Tij2,B_i)+SWRdir_i
        # Incoming radiation [W m^-2]
        A_i			=	B_i/ai
        for i in range(0,len(ai)):
            if ai[i] == 0:
                A_i[i] = A_i1[i]
        # Assumption: The sky has a fixed emission of LWR. Hence, Qnet is 0.
        A_i[5]		=	0
        # Absorbed shortwave radiation at ech surface Qnet_i
        Qnet_i		=	A_i-B_i


        # Assignment
        # Outgoing radiation [W m^-2]
        SWRout_i = B_i
        # Incoming radiation [W m^-2]
        SWRin_i	= A_i
        # Net absorbed radiation [W m^-2]
        SWRnet_i = Qnet_i

        # Energy balance
        SWRin_atm = SWR_dir + SWR_diff

        TotalSWRSurface_in = SWRin_i[0] * fgveg * A_g / A_g + SWRin_i[1] * fgbare * A_g / A_g + SWRin_i[2] * fgimp * A_g / A_g +\
                             SWRin_i[3] * A_w / A_g + SWRin_i[4] * A_w / A_g

        TotalSWRSurface_abs = SWRnet_i[0] * fgveg * A_g / A_g + SWRnet_i[1] * fgbare * A_g / A_g + SWRnet_i[2] * fgimp * A_g / A_g +\
                              SWRnet_i[3] * A_w / A_g + SWRnet_i[4] * A_w / A_g

        TotalSWRSurface_out = SWRout_i[0] * fgveg * A_g / A_s + SWRout_i[1] * fgbare * A_g / A_s + SWRout_i[2] * fgimp * A_g / A_s + \
                              SWRout_i[3] * A_w / A_s + SWRout_i[4] * A_w / A_s

        TotalSWRref_to_atm = SWRout_i[0] * F_sg_nT * fgveg + SWRout_i[1] * F_sg_nT * fgbare + SWRout_i[2] * F_sg_nT * fgimp +\
                             SWRout_i[3] * F_sw_nT + SWRout_i[4] * F_sw_nT

        # Incoming shortwave radiation
        class SWRin_nT_Def():
            pass
        SWRin_nT = SWRin_nT_Def()
        SWRin_nT.SWRinGroundImp = SWRin_i[2] * Cimp
        SWRin_nT.SWRinGroundBare = SWRin_i[1] * Cbare
        SWRin_nT.SWRinGroundVeg = SWRin_i[0] * Cveg
        SWRin_nT.SWRinTree = 0
        SWRin_nT.SWRinWallSun = SWRin_i[3]
        SWRin_nT.SWRinWallShade = SWRin_i[4]
        SWRin_nT.SWRinTotalGround = fgveg * SWRin_i[0] + fgbare * SWRin_i[1] + fgimp * SWRin_i[2]
        SWRin_nT.SWRinTotalCanyon = SWRin_i[0] * fgveg * A_g / A_g + SWRin_i[1] * fgbare * A_g / A_g + SWRin_i[2] * fgimp * A_g / A_g + \
                                    SWRin_i[3] * A_w / A_g + SWRin_i[4] * A_w / A_g

        # Outgoing shortwave radiation
        class SWRout_nT_Def():
            pass
        SWRout_nT = SWRout_nT_Def()
        SWRout_nT.SWRoutGroundImp = SWRout_i[2] * Cimp
        SWRout_nT.SWRoutGroundBare = SWRout_i[1] * Cbare
        SWRout_nT.SWRoutGroundVeg = SWRout_i[0] * Cveg
        SWRout_nT.SWRoutTree = 0
        SWRout_nT.SWRoutWallSun = SWRout_i[3]
        SWRout_nT.SWRoutWallShade = SWRout_i[4]
        SWRout_nT.SWRoutTotalGround = fgveg * SWRout_i[0] + fgbare * SWRout_i[1] + fgimp * SWRout_i[2]
        SWRout_nT.SWRoutTotalCanyon = SWRout_i[0] * fgveg * A_g / A_g + SWRout_i[1] * fgbare * A_g / A_g + SWRout_i[2] * fgimp * A_g / A_g + \
                                      SWRout_i[3] * A_w / A_g + SWRout_i[4] * A_w / A_g

        # Absorbed shortwave radiation
        class SWRabs_nT_Def():
            pass
        SWRabs_nT = SWRabs_nT_Def()
        SWRabs_nT.SWRabsGroundImp = SWRnet_i[2] * Cimp
        SWRabs_nT.SWRabsGroundBare = SWRnet_i[1] * Cbare
        SWRabs_nT.SWRabsGroundVeg = SWRnet_i[0] * Cveg
        SWRabs_nT.SWRabsTree = 0
        SWRabs_nT.SWRabsWallSun = SWRnet_i[3]
        SWRabs_nT.SWRabsWallShade = SWRnet_i[4]
        SWRabs_nT.SWRabsTotalGround = fgveg * SWRnet_i[0] + fgbare * SWRnet_i[1] + fgimp * SWRnet_i[2]
        SWRabs_nT.SWRabsTotalCanyon = SWRnet_i[0] * fgveg * A_g / A_g + SWRnet_i[1] * fgbare * A_g / A_g + SWRnet_i[2] * fgimp * A_g / A_g + \
                                      SWRnet_i[3] * A_w / A_g + SWRnet_i[4] * A_w / A_g

        # Direct absorbed shortwave radiation
        class SWRabsDir_nT_Def():
            pass
        SWRabsDir_nT = SWRabsDir_nT_Def()
        SWRabsDir_nT.SWRabsGroundImp = (1-agimp)*SWRdir_ground*Cimp
        SWRabsDir_nT.SWRabsGroundBare = (1-agbare)*SWRdir_ground*Cbare
        SWRabsDir_nT.SWRabsGroundVeg = (1-agveg)*SWRdir_ground*Cveg
        SWRabsDir_nT.SWRabsTree = 0
        SWRabsDir_nT.SWRabsWallSun = (1-aw)*SWRdir_wallsun
        SWRabsDir_nT.SWRabsWallShade = (1-aw)*SWRdir_wallshade
        SWRabsDir_nT.SWRabsTotalGround = fgveg*(1-agveg)*SWRdir_ground+fgbare*(1-agbare)*SWRdir_ground+fgimp*(1-agimp)*SWRdir_ground
        SWRabsDir_nT.SWRabsTotalCanyon = fgveg*(1-agveg)*SWRdir_ground*A_g/A_g+fgbare*(1-agbare)*SWRdir_ground*A_g/A_g+\
                                         fgimp*(1-agimp)*SWRdir_ground*A_g/A_g + (1-aw)*SWRdir_wallsun*A_w/A_g + (1-aw)*SWRdir_wallshade*A_w/A_g

        # Diffuse absorbed shortwave radiation
        class SWRabsDiff_nT_Def():
            pass
        SWRabsDiff_nT = SWRabsDiff_nT_Def()
        SWRabsDiff_nT.SWRabsGroundImp = (SWRabs_nT.SWRabsGroundImp-SWRabsDir_nT.SWRabsGroundImp)*Cimp
        SWRabsDiff_nT.SWRabsGroundBare = (SWRabs_nT.SWRabsGroundBare-SWRabsDir_nT.SWRabsGroundBare)*Cbare
        SWRabsDiff_nT.SWRabsGroundVeg = (SWRabs_nT.SWRabsGroundVeg-SWRabsDir_nT.SWRabsGroundVeg)*Cveg
        SWRabsDiff_nT.SWRabsTree = 0
        SWRabsDiff_nT.SWRabsWallSun = SWRabs_nT.SWRabsWallSun-SWRabsDir_nT.SWRabsWallSun
        SWRabsDiff_nT.SWRabsWallShade = SWRabs_nT.SWRabsWallShade-SWRabsDir_nT.SWRabsWallShade
        SWRabsDiff_nT.SWRabsTotalGround = SWRabs_nT.SWRabsTotalGround-SWRabsDir_nT.SWRabsTotalGround
        SWRabsDiff_nT.SWRabsTotalCanyon = SWRabs_nT.SWRabsTotalCanyon-SWRabsDir_nT.SWRabsTotalCanyon

        # Energy Balance of shortwave radiation
        class SWREB_nT_Def():
            pass
        SWREB_nT = SWREB_nT_Def()
        SWREB_nT.SWREBGroundImp = SWRin_nT.SWRinGroundImp - SWRout_nT.SWRoutGroundImp - SWRabs_nT.SWRabsGroundImp
        SWREB_nT.SWREBGroundBare = SWRin_nT.SWRinGroundBare - SWRout_nT.SWRoutGroundBare - SWRabs_nT.SWRabsGroundBare
        SWREB_nT.SWREBGroundVeg = SWRin_nT.SWRinGroundVeg - SWRout_nT.SWRoutGroundVeg - SWRabs_nT.SWRabsGroundVeg
        SWREB_nT.SWREBTree = 0
        SWREB_nT.SWREBWallSun = SWRin_nT.SWRinWallSun-SWRout_nT.SWRoutWallSun - SWRabs_nT.SWRabsWallSun
        SWREB_nT.SWREBWallShade = SWRin_nT.SWRinWallShade-SWRout_nT.SWRoutWallShade - SWRabs_nT.SWRabsWallShade
        SWREB_nT.SWREBTotalGround = SWRin_nT.SWRinTotalGround-SWRout_nT.SWRoutTotalGround - SWRabs_nT.SWRabsTotalGround
        SWREB_nT.SWREBTotalCanyon = SWRin_nT.SWRinTotalCanyon-SWRout_nT.SWRoutTotalCanyon - SWRabs_nT.SWRabsTotalCanyon

        if abs(SWREB_nT.SWREBGroundImp) >= 10**(-6):
            print('SWREB_nT.SWREBGroundImp is not 0')
        elif abs(SWREB_nT.SWREBGroundBare) >= 10**(-6):
            print('SWREB_nT.SWREBGroundBare is not 0')
        elif abs(SWREB_nT.SWREBGroundVeg) >= 10**(-6):
            print('SWREB_nT.SWREBGroundVeg	 is not 0')
        elif abs(SWREB_nT.SWREBWallSun) >= 10**(-6):
            print('SWREB_nT.SWREBWallSun is not 0')
        elif abs(SWREB_nT.SWREBWallShade) >= 10**(-6):
            print('SWREB_nT.SWREBWallShade is not 0')
        elif abs(SWREB_nT.SWREBTotalGround) >= 10**(-6):
            print('SWREB_nT.SWREBTotalGround is not ')
        elif abs(SWREB_nT.SWREBTotalCanyon) >= 10**(-6):
            print('SWREB_nT.SWREBTotalCanyon is not 0')

        return SWRin_nT,SWRout_nT,SWRabs_nT,SWRabsDir_nT,SWRabsDiff_nT,SWREB_nT

    def SWRabsorbedWithTrees(self,h_can,w_can,h_tree,r_tree,d_tree,fgveg,fgbare,fgimp,aw,agveg,agbare,agimp,at,LAIt,
                             SWR_dir,SWR_diff,theta_Z,theta_n,ViewFactor,ParVegTree):

        F_gs_T = ViewFactor.F_gs_T
        F_gt_T = ViewFactor.F_gt_T
        F_gw_T = ViewFactor.F_gw_T
        F_ww_T = ViewFactor.F_ww_T
        F_wt_T = ViewFactor.F_wt_T
        F_wg_T = ViewFactor.F_wg_T
        F_ws_T = ViewFactor.F_ws_T
        F_sg_T = ViewFactor.F_sg_T
        F_sw_T = ViewFactor.F_sw_T
        F_st_T = ViewFactor.F_st_T
        F_tg_T = ViewFactor.F_tg_T
        F_tw_T = ViewFactor.F_tw_T
        F_ts_T = ViewFactor.F_ts_T
        F_tt_T = ViewFactor.F_tt_T

        # normalized surface areas
        A_s = w_can
        A_g = w_can
        A_w = h_can
        # There are 2 trees. Hence, the area of tree is twice a circle
        A_t = 2 * 2 * numpy.pi * r_tree

        # load shortwave radiation
        SWRdir_ground, SWRdir_wallsun, SWRdir_wallshade, SWRdir_tree = self.DirectSWRSurfaces(h_can, w_can, d_tree, h_tree,
                                                                                              r_tree, theta_Z, theta_n,
                                                                                              SWR_dir, LAIt, 1,ParVegTree)

        SVF = numpy.zeros(4)
        SVF[0] = F_gs_T+F_gt_T+2*F_gw_T
        SVF[1] = F_ww_T+F_wt_T+F_wg_T+F_ws_T
        SVF[2] = F_sg_T+2*F_sw_T+F_st_T
        SVF[3] = F_ts_T+2*F_tw_T+F_tt_T+F_tg_T

        SVF2 = numpy.zeros(4)
        SVF2[0] = F_gs_T+2*F_ws_T*A_w+F_ts_T*A_t
        SVF2[1] = F_sg_T+2*F_wg_T*A_w+F_tg_T*A_t
        SVF2[2] = F_ww_T+F_sw_T*A_g/A_w+F_gw_T*A_g/A_w+F_tw_T*A_t/A_w
        SVF2[3] = F_gt_T*A_g/A_t+2*F_wt_T*A_w/A_t+F_tt_T

        if fgimp > 0:
            Cimp = 1
        else:
            Cimp = 0
        if fgbare > 0:
            Cbare = 1
        else:
            Cbare = 0
        if fgveg > 0:
            Cveg = 1
        else:
            Cveg = 0

        ai = [agveg, agbare, agimp, aw, aw, at, 0]

        # View factor matrix to solve for infinite reflections equation
        Tij = numpy.array([[1,0,0, -agveg*F_gw_T*Cveg, -agveg*F_gw_T*Cveg, -agveg*F_gt_T*Cveg, -agveg*F_gs_T*Cveg],
               [0,1,0, -agbare*F_gw_T*Cbare, -agbare*F_gw_T*Cbare, -agbare*F_gt_T*Cbare, -agbare*F_gs_T*Cbare],
               [0,0,1, -agimp*F_gw_T*Cimp, -agimp*F_gw_T*Cimp, -agimp*F_gt_T*Cimp, -agimp*F_gs_T*Cimp],
               [-aw*F_wg_T*fgveg*Cveg,-aw*F_wg_T*fgbare*Cbare,-aw*F_wg_T*fgimp*Cimp, 1, -aw*F_ww_T, -aw*F_wt_T, -aw*F_ws_T],
               [-aw*F_wg_T*fgveg*Cveg,-aw*F_wg_T*fgbare*Cbare,-aw*F_wg_T*fgimp*Cimp, -aw*F_ww_T, 1, -aw*F_wt_T, -aw*F_ws_T],
               [-at*F_tg_T*fgveg*Cveg,-at*F_tg_T*fgbare*Cbare,-at*F_tg_T*fgimp*Cimp, -at*F_tw_T, -at*F_tw_T, 1-at*F_tt_T, -at*F_ts_T],
               [0, 0, 0, 0, 0, 0, 1]])

        # Incoming shortwave radiation from sky
        Omega_i = numpy.array([agveg * SWRdir_ground * Cveg,
                   agbare * SWRdir_ground * Cbare,
                   agimp * SWRdir_ground * Cimp,
                   aw * SWRdir_wallsun,
                   aw * 0,
                   at * SWRdir_tree,
                   SWR_diff])

        # Outgoing radiation per surface
        B_i = numpy.linalg.solve(Tij,Omega_i)

        # Incoming shortwave radiation at each surface A_i
        Tij2 = numpy.array([[0, 0, 0, F_gw_T*Cveg, F_gw_T*Cveg, F_gt_T*Cveg, F_gs_T*Cveg],
                [0, 0, 0, F_gw_T*Cbare, F_gw_T*Cbare, F_gt_T*Cbare, F_gs_T*Cbare],
                [0, 0, 0, F_gw_T*Cimp, F_gw_T*Cimp, F_gt_T*Cimp, F_gs_T*Cimp],
                [F_wg_T*fgveg*Cveg, F_wg_T*fgbare*Cbare, F_wg_T*fgimp*Cimp, 0, F_ww_T, F_wt_T, F_ws_T],
                [F_wg_T*fgveg*Cveg, F_wg_T*fgbare*Cbare, F_wg_T*fgimp*Cimp, F_ww_T, 0, F_wt_T, F_ws_T],
                [F_tg_T*fgveg*Cveg, F_tg_T*fgbare*Cbare, F_tg_T*fgimp*Cimp, F_tw_T, F_tw_T, F_tt_T, F_ts_T],
                [0, 0, 0, 0, 0, 0, 0]])

        SWRdir_i = numpy.array([SWRdir_ground * Cveg,
                    SWRdir_ground * Cbare,
                    SWRdir_ground * Cimp,
                    SWRdir_wallsun,
                    0,
                    SWRdir_tree,
                    0])

        # Incoming radiation [W m^-2]
        A_i1 = numpy.dot(Tij2, B_i) + SWRdir_i
        # Incoming radiation [W m^-2]
        A_i = B_i / ai
        for i in range(0, len(ai)):
            if ai[i] == 0:
                A_i[i] = A_i1[i]
        A_i[6] = 0  # Assumption: The sky has a fixed emission of LWR. Hence, Qnet is 0.
        # Absorbed shortwave radiation at ech surface Qnet_i
        Qnet_i = A_i - B_i

        # Assignment
        # Outgoing radiation [W m^-2]
        SWRout_i = B_i
        # Incoming radiation [W m^-2]
        SWRin_i = A_i
        # Net absorbed radiation [W m^-2]
        SWRnet_i = Qnet_i

        # Energy balance
        SWRin_atm = SWR_dir + SWR_diff

        TotalSWRSurface_in = SWRin_i[0] * fgveg * A_g / A_g + SWRin_i[1] * fgbare * A_g / A_g + SWRin_i[2] * fgimp * A_g / A_g +\
                             SWRin_i[3] * A_w / A_g + SWRin_i[4] * A_w / A_g + SWRin_i[5]*A_t/A_g

        TotalSWRSurface_abs = SWRnet_i[0] * fgveg * A_g / A_g + SWRnet_i[1] * fgbare * A_g / A_g + SWRnet_i[2] * fgimp * A_g / A_g +\
                              SWRnet_i[3] * A_w / A_g + SWRnet_i[4] * A_w / A_g + SWRnet_i[5]*A_t/A_g

        TotalSWRSurface_out = SWRout_i[0] * fgveg * A_g / A_s + SWRout_i[1] * fgbare * A_g / A_s + SWRout_i[2] * fgimp * A_g / A_s + \
                              SWRout_i[3] * A_w / A_s + SWRout_i[4] * A_w / A_s + SWRout_i[5]*A_t/A_s

        TotalSWRref_to_atm = SWRout_i[0] * F_sg_T * fgveg + SWRout_i[1] * F_sg_T * fgbare + SWRout_i[2] * F_sg_T * fgimp +\
                             SWRout_i[3] * F_sw_T + SWRout_i[4] * F_sw_T + SWRout_i[5]*F_st_T

        TotalSWRref_to_atm2 = SWRout_i[0] * F_gs_T * fgveg + SWRout_i[1] * F_gs_T * fgbare + SWRout_i[2] * F_gs_T * fgimp +\
                             SWRout_i[3] * F_ws_T + SWRout_i[4] * F_ws_T + SWRout_i[5]*F_ts_T

        # Incoming shortwave radiation
        class SWRin_T_Def():
            pass
        SWRin_T = SWRin_T_Def()
        SWRin_T.SWRinGroundImp = SWRin_i[2] * Cimp
        SWRin_T.SWRinGroundBare = SWRin_i[1] * Cbare
        SWRin_T.SWRinGroundVeg = SWRin_i[0] * Cveg
        SWRin_T.SWRinTree = SWRin_i[5]
        SWRin_T.SWRinWallSun = SWRin_i[3]
        SWRin_T.SWRinWallShade = SWRin_i[4]
        SWRin_T.SWRinTotalGround = fgveg * SWRin_i[0] + fgbare * SWRin_i[1] + fgimp * SWRin_i[2]
        SWRin_T.SWRinTotalCanyon = SWRin_i[0] * fgveg * A_g / A_g + SWRin_i[1] * fgbare * A_g / A_g + SWRin_i[2] * fgimp * A_g / A_g + \
                                    SWRin_i[3] * A_w / A_g + SWRin_i[4] * A_w / A_g  +SWRin_i[5]*A_t/A_g

        # Outgoing shortwave radiation
        class SWRout_T_Def():
            pass
        SWRout_T = SWRout_T_Def()
        SWRout_T.SWRoutGroundImp = SWRout_i[2] * Cimp
        SWRout_T.SWRoutGroundBare = SWRout_i[1] * Cbare
        SWRout_T.SWRoutGroundVeg = SWRout_i[0] * Cveg
        SWRout_T.SWRoutTree = SWRout_i[5]
        SWRout_T.SWRoutWallSun = SWRout_i[3]
        SWRout_T.SWRoutWallShade = SWRout_i[4]
        SWRout_T.SWRoutTotalGround = fgveg * SWRout_i[0] + fgbare * SWRout_i[1] + fgimp * SWRout_i[2]
        SWRout_T.SWRoutTotalCanyon = SWRout_i[0] * fgveg * A_g / A_g + SWRout_i[1] * fgbare * A_g / A_g + SWRout_i[2] * fgimp * A_g / A_g + \
                                    SWRout_i[3] * A_w / A_g + SWRout_i[4] * A_w / A_g + SWRout_i[5]*A_t/A_g

        # Absorbed shortwave radiation
        class SWRabs_T_Def():
            pass
        SWRabs_T = SWRabs_T_Def()
        SWRabs_T.SWRabsGroundImp = SWRnet_i[2] * Cimp
        SWRabs_T.SWRabsGroundBare = SWRnet_i[1] * Cbare
        SWRabs_T.SWRabsGroundVeg = SWRnet_i[0] * Cveg
        SWRabs_T.SWRabsTree = SWRnet_i[5]
        SWRabs_T.SWRabsWallSun = SWRnet_i[3]
        SWRabs_T.SWRabsWallShade = SWRnet_i[4]
        SWRabs_T.SWRabsTotalGround = fgveg * SWRnet_i[0] + fgbare * SWRnet_i[1] + fgimp * SWRnet_i[2]
        SWRabs_T.SWRabsTotalCanyon = SWRnet_i[0] * fgveg * A_g / A_g + SWRnet_i[1] * fgbare * A_g / A_g + SWRnet_i[2] * fgimp * A_g / A_g + \
                                    SWRnet_i[3] * A_w / A_g + SWRnet_i[4] * A_w / A_g + SWRnet_i[5]*A_t/A_g

        # Direct absorbed shortwave radiation
        class SWRabsDir_T_Def():
            pass
        SWRabsDir_T = SWRabsDir_T_Def()
        SWRabsDir_T.SWRabsGroundImp = (1-agimp)*SWRdir_ground*Cimp
        SWRabsDir_T.SWRabsGroundBare = (1-agbare)*SWRdir_ground*Cbare
        SWRabsDir_T.SWRabsGroundVeg = (1-agveg)*SWRdir_ground*Cveg
        SWRabsDir_T.SWRabsTree = (1-at)*SWRdir_tree
        SWRabsDir_T.SWRabsWallSun = (1-aw)*SWRdir_wallsun
        SWRabsDir_T.SWRabsWallShade = (1-aw)*SWRdir_wallshade
        SWRabsDir_T.SWRabsTotalGround = fgveg*(1-agveg)*SWRdir_ground+fgbare*(1-agbare)*SWRdir_ground+fgimp*(1-agimp)*SWRdir_ground
        SWRabsDir_T.SWRabsTotalCanyon = fgveg*(1-agveg)*SWRdir_ground*A_g/A_g+fgbare*(1-agbare)*SWRdir_ground*A_g/A_g+\
                                        fgimp*(1-agimp)*SWRdir_ground*A_g/A_g + (1-aw)*SWRdir_wallsun*A_w/A_g + (1-aw)*SWRdir_wallshade*A_w/A_g + \
                                        (1-at)*SWRdir_tree*A_t/A_g

        # Diffuse absorbed shortwave radiation
        class SWRabsDiff_T_Def():
            pass
        SWRabsDiff_T = SWRabsDiff_T_Def()
        SWRabsDiff_T.SWRabsGroundImp = (SWRabs_T.SWRabsGroundImp-SWRabsDir_T.SWRabsGroundImp)*Cimp
        SWRabsDiff_T.SWRabsGroundBare = (SWRabs_T.SWRabsGroundBare-SWRabsDir_T.SWRabsGroundBare)*Cbare
        SWRabsDiff_T.SWRabsGroundVeg = (SWRabs_T.SWRabsGroundVeg-SWRabsDir_T.SWRabsGroundVeg)*Cveg
        SWRabsDiff_T.SWRabsTree = SWRabs_T.SWRabsTree-SWRabsDir_T.SWRabsTree
        SWRabsDiff_T.SWRabsWallSun = SWRabs_T.SWRabsWallSun-SWRabsDir_T.SWRabsWallSun
        SWRabsDiff_T.SWRabsWallShade = SWRabs_T.SWRabsWallShade-SWRabsDir_T.SWRabsWallShade
        SWRabsDiff_T.SWRabsTotalGround = SWRabs_T.SWRabsTotalGround-SWRabsDir_T.SWRabsTotalGround
        SWRabsDiff_T.SWRabsTotalCanyon = SWRabs_T.SWRabsTotalCanyon-SWRabsDir_T.SWRabsTotalCanyon

        # Energy Balance of shortwave radiation
        class SWREB_T_Def():
            pass
        SWREB_T = SWREB_T_Def()
        SWREB_T.SWREBGroundImp = SWRin_T.SWRinGroundImp - SWRout_T.SWRoutGroundImp - SWRabs_T.SWRabsGroundImp
        SWREB_T.SWREBGroundBare = SWRin_T.SWRinGroundBare - SWRout_T.SWRoutGroundBare - SWRabs_T.SWRabsGroundBare
        SWREB_T.SWREBGroundVeg = SWRin_T.SWRinGroundVeg - SWRout_T.SWRoutGroundVeg - SWRabs_T.SWRabsGroundVeg
        SWREB_T.SWREBTree = SWRin_T.SWRinTree - SWRout_T.SWRoutTree - SWRabs_T.SWRabsTree
        SWREB_T.SWREBWallSun = SWRin_T.SWRinWallSun-SWRout_T.SWRoutWallSun - SWRabs_T.SWRabsWallSun
        SWREB_T.SWREBWallShade = SWRin_T.SWRinWallShade-SWRout_T.SWRoutWallShade - SWRabs_T.SWRabsWallShade
        SWREB_T.SWREBTotalGround = SWRin_T.SWRinTotalGround-SWRout_T.SWRoutTotalGround - SWRabs_T.SWRabsTotalGround
        SWREB_T.SWREBTotalCanyon = SWRin_T.SWRinTotalCanyon-SWRout_T.SWRoutTotalCanyon - SWRabs_T.SWRabsTotalCanyon

        if abs(SWREB_T.SWREBGroundImp) >= 10**(-6):
            print('SWREB_T.SWREBGroundImp is not 0')
        elif abs(SWREB_T.SWREBGroundBare) >= 10**(-6):
            print('SWREB_T.SWREBGroundBare is not 0')
        elif abs(SWREB_T.SWREBGroundVeg) >= 10**(-6):
            print('SWREB_T.SWREBGroundVeg	 is not 0')
        elif abs(SWREB_T.SWREBWallSun) >= 10**(-6):
            print('SWREB_T.SWREBWallSun is not 0')
        elif abs(SWREB_T.SWREBWallShade) >= 10**(-6):
            print('SWREB_T.SWREBWallShade is not 0')
        elif abs(SWREB_T.SWREBTotalGround) >= 10**(-6):
            print('SWREB_T.SWREBTotalGround is not ')
        elif abs(SWREB_T.SWREBTotalCanyon) >= 10**(-6):
            print('SWREB_T.SWREBTotalCanyon is not 0')
        elif abs(SWREB_T.SWREBTree) >= 10**(-6):
            print('SWREB_T.SWREBTree is not 0')

        return SWRin_T,SWRout_T,SWRabs_T,SWRabsDir_T,SWRabsDiff_T,SWREB_T


    def DirectSWRSurfaces(self,h_can,w_can,d_tree,h_tree,r_tree,theta_Z,theta_n,SWR_dir,LAIt,trees,ParVegTree):

        Kopt_T = ParVegTree.Kopt

        # Calculation of SWRdir_t
        if trees == 0:
            tau = 0
            SWRdir_t = 0
        else:
            SWR_tree1, SWR_tree2 = self.DirectSWRTrees(h_can,d_tree,h_tree,r_tree,theta_Z,theta_n,SWR_dir)
            # Calculate how much shortwave radiation passes through the trees
            tau = numpy.exp(-Kopt_T * LAIt)
            # averaging over the two trees
            SWRdir_t = (1 - tau) * (SWR_tree1 + SWR_tree2) / 2

        # Calculation of SWRdir_g, SWRdir_wsun, SWRdir_wshd
        if trees == 0:
            X_Shadow, X_Tree, n_Shadow, n_Tree = self.ShadowLengthNoTree(h_can, w_can, theta_Z, theta_n)
        else:
            X_Shadow, X_Tree, n_Shadow, n_Tree = self.ShadowLengthWithTrees(h_can, w_can, d_tree, h_tree, r_tree, theta_Z, theta_n)

        Xsi = math.tan(theta_Z) * abs(math.sin(theta_n)) ### check later

        SWRdir_g = SWR_dir * (1 - X_Shadow + tau * X_Tree)
        SWRdir_wsun = SWR_dir * Xsi * (1 - n_Shadow + tau * n_Tree)
        SWRdir_wshd = 0

        A_g = w_can
        A_w = h_can
        A_t = 2 * numpy.pi * r_tree

        check_total = A_g / A_g * SWRdir_g + A_w / A_g * SWRdir_wsun + A_w / A_g * SWRdir_wshd + 2 * A_t / A_g * SWRdir_t

        if abs(check_total - SWR_dir) > 1e-10:
            delta = check_total - SWR_dir
            if delta < 0:
                # the energy excess or deficit is distributed to the trees
                SWRdir_t = SWRdir_t - delta * A_g / (2 * A_t)
            else:
                # the energy excess or deficit is distributed to the trees
                SWRdir_t = SWRdir_t - delta * A_g / (2 * A_t)

        return SWRdir_g,SWRdir_wsun,SWRdir_wshd,SWRdir_t


    def DirectSWRTrees(self,h_can,d_tree,h_tree,r_tree,theta_Z,theta_n,SWR_dir):

        Xsi = math.tan(theta_Z) * abs(math.sin(theta_n))

        tan_theta1 = ((1-d_tree)*(h_can-h_tree)+r_tree*numpy.sqrt((1-d_tree)**2+(h_can-h_tree)**2-r_tree**2))/((h_can-h_tree)**2-r_tree**2)
        tan_theta2 = ((1-d_tree)*(h_can-h_tree)-r_tree*numpy.sqrt((1-d_tree)**2+(h_can-h_tree)**2-r_tree**2))/((h_can-h_tree)**2-r_tree**2)
        tan_theta3 = (d_tree*(h_can-h_tree)+r_tree*numpy.sqrt(d_tree**2+(h_can-h_tree)**2-r_tree**2))/((h_can-h_tree)**2-r_tree**2)
        tan_theta4 = (d_tree*(h_can-h_tree)-r_tree*numpy.sqrt(d_tree**2+(h_can-h_tree)**2-r_tree**2))/((h_can-h_tree)**2-r_tree**2)

        if Xsi >= tan_theta1:
            # Tree 1 is completely shaded
            SWR_tree1 = 0
        elif Xsi < tan_theta1 and Xsi >= tan_theta2:
            # Tree 1 is partially sunlit
            SWR_tree1 = SWR_dir*(r_tree*numpy.sqrt(1+Xsi**2) + (1 - d_tree) - (h_can - h_tree) * Xsi) / (2 * numpy.pi * r_tree)
        elif Xsi < tan_theta2:
            # tree 1 is completely sunlit
            SWR_tree1 = SWR_dir * (2 * r_tree * numpy.sqrt(1 + Xsi**2)) / (2 * numpy.pi * r_tree)
        else:
            # Account for weird angles at night (angles = NaN)
            SWR_tree1 = 0

        if Xsi >= tan_theta3:
            # Tree 2 is completely shaded
            SWR_tree2 = 0
        elif Xsi < tan_theta3 and Xsi >= tan_theta4:
            # Tree 2 is partially sunlit
            SWR_tree2 = SWR_dir * (r_tree * numpy.sqrt(1 + Xsi**2) + d_tree - (h_can - h_tree) * Xsi) / (2 * numpy.pi * r_tree)
        elif Xsi<tan_theta4:
            # tree 1 is completely sunlit
            SWR_tree2 = SWR_dir * (2 * r_tree * numpy.sqrt(1 + Xsi**2)) / (2 * numpy.pi * r_tree)
        else:
            # Account for weird angles at night (angles = NaN)
            SWR_tree2 = 0

        return SWR_tree1,SWR_tree2

    def ShadowLengthNoTree(self,h_can,w_can,theta_Z,theta_n):

        Xsi = math.tan(theta_Z) * abs(math.sin(theta_n))

        # Shadow by the Wall
        # shadow cast on the ground by the wall
        X_Shadow = h_can * Xsi
        # shadow cast on the opposite wall by the wall
        n_Shadow = h_can - w_can / Xsi

        if abs(X_Shadow) < w_can:
            n_Shadow = 0
        else:
            X_Shadow = w_can

        if n_Shadow < h_can:
            n_Shadow = n_Shadow
        else:
            n_Shadow = h_can

        # NOTE : the origin (0,0) is the lower left corner of the canyon
        X_Shadow = X_Shadow
        X_tree = 0
        n_Shadow = n_Shadow / h_can
        n_tree = 0

        return X_Shadow,X_tree,n_Shadow,n_tree


    def ShadowLengthWithTrees(self,h_can,w_can,d_tree,h_tree,r_tree,theta_Z,theta_n):

        Xsi = math.tan(theta_Z) * abs(math.sin(theta_n))

        # Shadow by the Wall
        X_wall = h_can * Xsi
        n_wall = h_can - w_can / Xsi

        if abs(X_wall) < w_can:
            n_wall = 0
        else:
            X_wall = w_can

        x0 = max(0., w_can - h_can * Xsi)
        y0 = max(0., h_can - w_can / Xsi)
        secXsi = numpy.sqrt(1 + Xsi ** 2)
        cosecXsi = numpy.sqrt(1 + 1 / Xsi ** 2)

        # Shadow by the Tree 1
        x1 = max(0, d_tree - h_tree * Xsi - r_tree * secXsi)
        y1 = max(0, h_tree - (w_can - d_tree) / Xsi - r_tree * cosecXsi)
        x2 = max(0, d_tree - h_tree * Xsi + r_tree * secXsi)
        y2 = max(0, h_tree - (w_can - d_tree) / Xsi + r_tree * cosecXsi)

        X_Tree1 = x2 - x1

        # Shadow by the Tree 2
        x3 = max(0, w_can - d_tree - h_tree * Xsi - r_tree * secXsi)
        y3 = max(0, h_tree - d_tree / Xsi - r_tree * cosecXsi)
        x4 = max(0, w_can - d_tree - h_tree * Xsi + r_tree * secXsi)
        y4 = max(0, h_tree - d_tree / Xsi + r_tree * cosecXsi)

        X_Tree2 = x4 - x3
        n_Tree1 = y4 - y3
        n_Tree2 = y2 - y1

        # Total shadow length on the ground by wall, Tree 1, and Tree 2
        delta = max(0, x2 - x0)

        if x0 < x4:
            X_Shadow = w_can - min(x0,x3) + X_Tree1 - delta
            if x0 < x3:
                X_Tree = X_Tree1 - delta
            else:
                X_Tree = X_Tree1 + x0 - x3
        elif x0 >= x4:
            X_Shadow = X_wall + X_Tree1 + X_Tree2
            X_Tree = X_Tree1 + X_Tree2

        # Total shadow length on the wall by wall, Tree 1, and Tree 2
        lowest_shaded = max(y0,y2)
        if y3 > lowest_shaded:
            n_Shadow = n_Tree1 + lowest_shaded
            if y2 > n_wall:
                n_Tree = n_Tree1 + y2 - n_wall
            elif y2 <= n_wall:
                n_Tree = n_Tree1
            elif y1 > n_wall:
                n_Tree = n_Tree1 + n_Tree2
        else:
            n_Shadow = max([y0,y1,y2,y3,y4])
            if (y4 > n_wall):
                n_Tree = y4 - n_wall
            else:
                n_Tree = 0.

        n_Tree = n_Tree / h_can
        n_Shadow = n_Shadow / h_can

        return X_Shadow,X_Tree,n_Shadow,n_Tree

    def TotalSWR_LWR_Rrual(self,RSMParam,layerTemp_Rural,MeteoData,SunPosition,simTime):

        # Stefan Boltzmann constant [W m^-2 K^-4]
        SIGMA = 5.67e-08
        # Calculate outgoing and net longwave radiation in the rural area  [W m^-2]
        # Outgoing longwave radiation [W m^-2]
        L_rural_emt = RSMParam.e_rural * SIGMA * layerTemp_Rural[0]**4
        # Incoming longwave radiation [W m^-2]
        L_rural_in = RSMParam.e_rural * MeteoData.LWR
        # Net longwave radiation at the rural surface [W m^-2]
        rural_infra = L_rural_in - L_rural_emt

        # Calculate incoming and net shortwave in the rural
        SDir_rural = max(math.cos(SunPosition.theta_Z) * MeteoData.SW_dir, 0.0)
        # Winter: no vegetation
        if simTime.month < RSMParam.vegStart or simTime.month > RSMParam.vegEnd:
            S_rural = (1 - RSMParam.a_rural) * (SDir_rural + MeteoData.SW_diff)
            S_rural_out = RSMParam.a_rural * (SDir_rural + MeteoData.SW_diff)
        # Summer: effect of vegetation is considered
        else:
            S_rural = ((1-RSMParam.rurVegCover)*(1-RSMParam.a_rural)+RSMParam.rurVegCover*(1-RSMParam.aveg_rural))*(SDir_rural+MeteoData.SW_diff)
            S_rural_out = ((1-RSMParam.rurVegCover)*RSMParam.a_rural+RSMParam.rurVegCover*RSMParam.aveg_rural)*(SDir_rural+MeteoData.SW_diff)


        rural_solRec = SDir_rural + MeteoData.SW_diff
        rural_solAbs = S_rural

        class SWR_Rural_Def():
            pass
        SWR_Rural = SWR_Rural_Def()
        SWR_Rural.SWRinRural = rural_solRec
        SWR_Rural.SWRoutRural = S_rural_out
        SWR_Rural.SWRabsRural = rural_solAbs
        class LWR_Rural_Def():
            pass
        LWR_Rural = LWR_Rural_Def()
        LWR_Rural.LWRinRural = L_rural_in
        LWR_Rural.LWRoutRural = L_rural_emt
        LWR_Rural.LWRabsRural = rural_infra

        return SWR_Rural,LWR_Rural
















