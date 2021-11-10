import os
import numpy
import math
from pprint import pprint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import copy

'''
Ray tracing:
Developed by Mohsen Moradi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Originally developed by Naika Meili
Last update: January 2021
'''

class RayTracingCal(object):

    def VFUrbanCanyon(self,OPTION_RAY,Name_Ray,Gemeotry_m,geometry,Person,ParTree,ViewFactor_file):

        # The sample size and the number of rays determine the run time. They must be carefully selected.
        MCSampleSize = 500
        NRays = 200

        if OPTION_RAY == 1:

            ViewFactor_file = numpy.loadtxt(ViewFactor_file)

            F_gs_nT = ViewFactor_file[0]
            F_gw_nT = ViewFactor_file[1]
            F_ww_nT = ViewFactor_file[2]
            F_wg_nT = ViewFactor_file[3]
            F_ws_nT = ViewFactor_file[4]
            F_sg_nT = ViewFactor_file[5]
            F_sw_nT = ViewFactor_file[6]
            F_gs_T = ViewFactor_file[7]
            F_gt_T = ViewFactor_file[8]
            F_gw_T = ViewFactor_file[9]
            F_ww_T = ViewFactor_file[10]
            F_wt_T = ViewFactor_file[11]
            F_wg_T = ViewFactor_file[12]
            F_ws_T = ViewFactor_file[13]
            F_sg_T = ViewFactor_file[14]
            F_sw_T = ViewFactor_file[15]
            F_st_T = ViewFactor_file[16]
            F_tg_T = ViewFactor_file[17]
            F_tw_T = ViewFactor_file[18]
            F_ts_T = ViewFactor_file[19]
            F_tt_T = ViewFactor_file[20]

            Sum_g = F_gs_T + F_gt_T + 2 * F_gw_T
            Sum_w = F_ww_T + F_wt_T + F_wg_T + F_ws_T
            Sum_s = F_sg_T + 2 * F_sw_T + F_st_T
            Sum_t = F_ts_T + 2 * F_tw_T + F_tt_T + F_tg_T

            F_pg = ViewFactor_file[21]
            F_ps = ViewFactor_file[22]
            F_pt = ViewFactor_file[23]
            F_pw = ViewFactor_file[24]

        else:

            if ParTree.trees == 0:
                geometry.radius_tree = 0
                geometry.htree = -Gemeotry_m.Height_canyon / 10000
                geometry.distance_tree = 0

            # compute view factors with monte carlo ray tracing
            F_gs_T, F_gt_T, F_gw_T, F_ww_T, F_wt_T, F_wg_T, F_ws_T, F_ts_T, F_tw_T, F_tt_T, F_tg_T, F_sg_T, F_sw_T, F_st_T,\
            F_pg, F_ps, F_pw, F_pt, VFRayTracingRaw_T, VFRayTracing_T = \
                self.VFRayTracingReciprocity(Gemeotry_m.Height_canyon, Gemeotry_m.Width_canyon, geometry.radius_tree,geometry.htree,
                                             geometry.distance_tree, Person, MCSampleSize, NRays)

        # calculate view factors with analytical solutions
        F_gs_nT, F_gt_nT, F_gw_nT, F_ww_nT, F_wt_nT, F_wg_nT, F_ws_nT, F_ts_nT, F_tw_nT, F_tt_nT, F_tg_nT, F_sg_nT, \
        F_sw_nT,F_st_nT, ViewFactor_nT = self.VFAnalytical(Gemeotry_m.Height_canyon, Gemeotry_m.Width_canyon)

        class ViewFactor_Def():
            pass
        ViewFactor = ViewFactor_Def()
        ViewFactor.F_gs_nT = F_gs_nT
        ViewFactor.F_gw_nT = F_gw_nT
        ViewFactor.F_ww_nT = F_ww_nT
        ViewFactor.F_wg_nT = F_wg_nT
        ViewFactor.F_ws_nT = F_ws_nT
        ViewFactor.F_sg_nT = F_sg_nT
        ViewFactor.F_sw_nT = F_sw_nT
        ViewFactor.F_gs_T = F_gs_T
        ViewFactor.F_gt_T = F_gt_T
        ViewFactor.F_gw_T = F_gw_T
        ViewFactor.F_ww_T = F_ww_T
        ViewFactor.F_wt_T = F_wt_T
        ViewFactor.F_wg_T = F_wg_T
        ViewFactor.F_ws_T = F_ws_T
        ViewFactor.F_sg_T = F_sg_T
        ViewFactor.F_sw_T = F_sw_T
        ViewFactor.F_st_T = F_st_T
        ViewFactor.F_tg_T = F_tg_T
        ViewFactor.F_tw_T = F_tw_T
        ViewFactor.F_ts_T = F_ts_T
        ViewFactor.F_tt_T = F_tt_T

        class ViewFactorPoint_Def():
            pass
        ViewFactorPoint = ViewFactorPoint_Def()
        ViewFactorPoint.F_pg = F_pg
        ViewFactorPoint.F_ps = F_ps
        ViewFactorPoint.F_pt = F_pt
        ViewFactorPoint.F_pw = F_pw

        return  ViewFactor,ViewFactorPoint

    def VFRayTracingReciprocity(self,H, W, a, ht, d, Person, MCSampleSize, NRays):

        _F_gs_T_, _F_gt_T_, _F_gw_T_, _F_ww_T_, _F_wt_T_, _F_wg_T_, _F_ws_T_, _F_ts_T_, _F_tw_T_, _F_tt_T_, _F_tg_T_, \
        _F_sg_T_, _F_sw_T_, _F_st_T_, F_pg, F_ps, F_pw, F_pt, VFRayTracingRaw_T = \
            self.VFRayTracing(H, W, a, ht, d, Person, MCSampleSize, NRays)

        h = H / W
        w = W / W
        ratio = h / w

        Sum = numpy.zeros(4)
        Sum2 = numpy.zeros(4)

        if a == 0:

            # The view factor taken from the ray tracing is F_gs_T
            F_gs_T = VFRayTracingRaw_T.F_gs_T
            # factor 0.5 because there are 2 walls that are seen by the ground
            F_gw_T = 0.5 * (1 - F_gs_T)
            F_gt_T = 0

            F_sg_T = F_gs_T * w / w
            F_sw_T = F_gw_T * w / w
            F_st_T = 0

            F_wg_T = F_gw_T * w / h
            F_ws_T = F_sw_T * w / h
            F_ww_T = 1 - F_wg_T - F_ws_T
            F_wt_T = 0

            F_tg_T = 0
            F_ts_T = 0
            F_tw_T = 0
            F_tt_T = 0

            Sum[0] = F_gs_T + 2 * F_gw_T
            Sum[1] = F_ww_T + F_wg_T + F_ws_T
            Sum[2] = F_sg_T + 2 * F_sw_T
            Sum[3] = 0

            Sum2[0] = F_sg_T * w / w + 2 * F_wg_T * h / w
            Sum2[1] = F_ww_T * h / h + F_gw_T * w / h + F_sw_T * w / h
            Sum2[2] = F_gs_T * w / w + 2 * F_ws_T * h / w
            Sum2[3] = 0

        else:
            # The view factors taken from the ray tracing are F_st_T, F_gs_T,F_gt_T, F_wt_T
            Atree = 2 * 2 * numpy.pi * a

            F_gs_T = VFRayTracingRaw_T.F_gs_T
            F_gt_T = VFRayTracingRaw_T.F_gt_T
            # factor 0.5 because there are 2 walls that are seen by the ground
            F_gw_T = 0.5 * (1 - F_gs_T - F_gt_T)

            F_sg_T = F_gs_T * w / w
            F_st_T = VFRayTracingRaw_T.F_st_T
            # factor 0.5 because there are 2 walls that are seen by the ground
            F_sw_T = 0.5 * (1 - F_sg_T - F_st_T)

            F_wg_T = F_gw_T * w / h
            F_ws_T = F_sw_T * w / h
            F_wt_T = VFRayTracingRaw_T.F_wt_T
            F_ww_T = 1 - F_wg_T - F_ws_T - F_wt_T

            F_ts_T = F_st_T * w / Atree
            F_tw_T = F_wt_T * h / Atree
            F_tg_T = F_gt_T * w / Atree
            F_tt_T = 1 - F_ts_T - 2 * F_tw_T - F_tg_T

            Sum[0] = F_gs_T + 2 * F_gw_T + F_gt_T
            Sum[1] = F_ww_T + F_wg_T + F_ws_T + F_wt_T
            Sum[2] = F_sg_T + 2 * F_sw_T + F_st_T
            Sum[3] = F_tg_T + 2 * F_tw_T + F_ts_T + F_tt_T

            Sum2[0] = F_sg_T * w / w + 2 * F_wg_T * h / w + F_tg_T * Atree / w
            Sum2[1] = F_ww_T * h / h + F_gw_T * w / h + F_sw_T * w / h + F_tw_T * Atree / h
            Sum2[2] = F_gs_T * w / w + 2 * F_ws_T * h / w + F_ts_T * Atree / w
            Sum2[3] = F_gt_T * w / Atree + 2 * F_wt_T * h / Atree + F_st_T * w / Atree + F_tt_T * Atree / Atree


        # Check sum
        if a > 0 and any(Sum < 0.9999) or any(Sum > 1.0001): ### Check any
            print('The view factor do not add up to 1. Please check the ray tracing algorithm.')
        elif a == 0 and any(Sum[0:3] < 0.9999) or any(Sum[0:3] > 1.0001):
            print('The view factor do not add up to 1. Please check the ray tracing algorithm.')

        # Assign view factors to struct
        class VFRayTracing_T_Def():
            pass
        VFRayTracing_T = VFRayTracing_T_Def()
        VFRayTracing_T.F_gs_T = F_gs_T
        VFRayTracing_T.F_gt_T = F_gt_T
        VFRayTracing_T.F_gw_T = F_gw_T
        VFRayTracing_T.F_ww_T = F_ww_T
        VFRayTracing_T.F_wt_T = F_wt_T
        VFRayTracing_T.F_wg_T = F_wg_T
        VFRayTracing_T.F_ws_T = F_ws_T
        VFRayTracing_T.F_sg_T = F_sg_T
        VFRayTracing_T.F_sw_T = F_sw_T
        VFRayTracing_T.F_st_T = F_st_T
        VFRayTracing_T.F_tg_T = F_tg_T
        VFRayTracing_T.F_tw_T = F_tw_T
        VFRayTracing_T.F_ts_T = F_ts_T
        VFRayTracing_T.F_tt_T = F_tt_T
        VFRayTracing_T.F_pg = F_pg
        VFRayTracing_T.F_ps = F_ps
        VFRayTracing_T.F_pw = F_pw
        VFRayTracing_T.F_pt = F_pt

        return F_gs_T,F_gt_T,F_gw_T,F_ww_T,F_wt_T,F_wg_T,F_ws_T,F_ts_T,F_tw_T,F_tt_T,F_tg_T,F_sg_T,F_sw_T,F_st_T,F_pg,\
               F_ps,F_pw,F_pt, VFRayTracingRaw_T,VFRayTracing_T

    def VFAnalytical(self,H,W):

        # Sky view factors without trees: Harman et al. 2004
        ratio = H / W

        F_gs_nT = numpy.sqrt(1 + (ratio) ** 2) - ratio
        F_gt_nT = 0
        # factor 0.5 because there are 2 walls that are seen by the ground
        F_gw_nT = 0.5 * (1 - F_gs_nT)

        F_ww_nT = numpy.sqrt(1 + (1 / ratio) ** 2) - 1 / ratio
        F_wt_nT = 0
        F_wg_nT = 0.5 * (1 - F_ww_nT)
        F_ws_nT = 0.5 * (1 - F_ww_nT)

        F_ts_nT = 0
        F_tw_nT = 0
        F_tt_nT = 0
        F_tg_nT = 0

        F_sg_nT = F_gs_nT
        F_sw_nT = ratio * F_ws_nT
        F_st_nT = 0

        # Check for unity of the sum of the view factors
        h = H / W
        w = W / W

        Sum_g = F_gs_nT + F_gt_nT + F_gw_nT * 2
        Sum_w = F_ww_nT + F_wt_nT + F_wg_nT + F_ws_nT
        Sum_t = F_ts_nT + 2 * F_tw_nT + F_tt_nT + F_tg_nT
        Sum_s = F_sg_nT + 2 * F_sw_nT + F_st_nT

        Sum_g2 = F_wg_nT * h / w * 2 + F_sg_nT * w / w
        Sum_w2 = F_gw_nT * w / h + F_ww_nT * h / h + F_sw_nT * w / h
        Sum_t2 = 0
        Sum_s2 = F_gs_nT * w / w + 2 * F_ws_nT * h / w

        class ViewFactor_nT_Def():
            pass
        ViewFactor_nT = ViewFactor_nT_Def()
        ViewFactor_nT.F_gs_nT = F_gs_nT
        ViewFactor_nT.F_gw_nT = F_gw_nT
        ViewFactor_nT.F_ww_nT = F_ww_nT
        ViewFactor_nT.F_wg_nT = F_wg_nT
        ViewFactor_nT.F_ws_nT = F_ws_nT
        ViewFactor_nT.F_sg_nT = F_sg_nT
        ViewFactor_nT.F_sw_nT = F_sw_nT

        return F_gs_nT,F_gt_nT,F_gw_nT,F_ww_nT,F_wt_nT,F_wg_nT,F_ws_nT,F_ts_nT,F_tw_nT,F_tt_nT,F_tg_nT,F_sg_nT,F_sw_nT,\
               F_st_nT,ViewFactor_nT

    def VFRayTracing(self,H,W,a,ht,d,Person,MCSampleSize,NRays):

        # Emitting surface
        # 1 = from wall 1
        # 2 = from wall 2
        # 3 = from ground
        # 4 = from tree 1
        # 5 = from tree 2
        # 6 = from sky
        # 7 = from point p

        View_factor = numpy.zeros((7,6))
        for option_surface in range(7):
            VG, VW1, VW2, VS, VT1, VT2 = self.View_Factors_Geometry(H, W, a, ht, d, Person,option_surface, MCSampleSize,NRays)
            # towards wall 1
            View_factor[option_surface, 0] = VW1
            # towards wall 2
            View_factor[option_surface, 1] = VW2
            # towards ground
            View_factor[option_surface, 2] = VG
            # towards tree 1
            View_factor[option_surface, 3] = VT1
            # towards tree 2
            View_factor[option_surface, 4] = VT2
            # towards sky
            View_factor[option_surface, 5] = VS


        # Elimination of self-view factor and rescaling
        # Wall 1 to wall 1 self view factor elimination
        View_factor[0,:] = [View_factor[0,i]/ (1 - View_factor[0,0]) for i in range(6)]
        View_factor[0,0] = 0
        # Wall 2 to wall 2 self view factor elimination
        View_factor[1,:] = [View_factor[1,i] / (1 - View_factor[1,1]) for i in range(6)]
        View_factor[1,1] = 0
        # Ground to Ground self view factor elimination
        View_factor[2, :] = [View_factor[2, i] / (1 - View_factor[2, 2]) for i in range(6)]
        View_factor[2, 2] = 0
        # Tree 1 to tree 1 self view factor elimination
        View_factor[3, :] = [View_factor[3, i] / (1 - View_factor[3, 3]) for i in range(6)]
        View_factor[3, 3] = 0
        # Tree 2 to tree 2 self view factor elimination
        View_factor[4, :] = [View_factor[4, i] / (1 - View_factor[4, 4]) for i in range(6)]
        View_factor[4, 4] = 0
        # Sky to Sky self view factor elimination
        View_factor[5, :] = [View_factor[5, i] / (1 - View_factor[5, 5]) for i in range(6)]
        View_factor[5, 5] = 0

        # View factor assignment
        F_gs_T = View_factor[2,5]
        F_gt_T = View_factor[2,3] + View_factor[2,4]
        F_gw_T = (View_factor[2,0] + View_factor[2,1]) / 2

        F_ww_T = View_factor[0,1]
        F_wt_T = View_factor[0,3] + View_factor[0,4]
        F_wg_T = View_factor[0,2]
        F_ws_T = View_factor[0,5]

        if a > 0:
            F_ts_T = View_factor[3,5]
            F_tw_T = (View_factor[3,0] + View_factor[3,1]) / 2
            F_tt_T = View_factor[3,4]
            F_tg_T = View_factor[3,2]
        else:
            F_ts_T = 0
            F_tw_T = 0
            F_tt_T = 0
            F_tg_T = 0

        F_sg_T = View_factor[5,2]
        F_sw_T = (View_factor[5,0] + View_factor[5,1]) / 2
        F_st_T = View_factor[5,3] + View_factor[5,4]

        F_pg = View_factor[6,2]
        F_ps = View_factor[6,5]
        F_pw = (View_factor[6,0] + View_factor[6,1]) / 2
        F_pt = View_factor[6,3] + View_factor[6,4]

        # Check sum
        Sum = numpy.zeros(5)
        Sum[0] = F_gs_T + F_gt_T + F_gw_T * 2
        Sum[1] = F_ww_T + F_wt_T + F_wg_T + F_ws_T
        Sum[2] = F_sg_T + 2 * F_sw_T + F_st_T
        Sum[3] = F_pg + F_ps + 2 * F_pw + F_pt
        Sum[4] = F_ts_T + 2 * F_tw_T + F_tt_T + F_tg_T

        if a == 0:
            F_ts_T = 0
            F_tw_T = 0
            F_tt_T = 0
            F_tg_T = 0

        if a > 0 and any(Sum < 0.9999) or any(Sum > 1.0001):
            print('The view factor do not add up to 1. Please check the ray tracing algorithm.')
        elif a == 0 and any(Sum[0:4] < 0.9999) and any(Sum[0:4] > 1.0001):
            print('The view factor do not add up to 1. Please check the ray tracing algorithm.')

        class VFRayTracingRaw_T_Def():
            pass
        VFRayTracingRaw_T = VFRayTracingRaw_T_Def()
        VFRayTracingRaw_T.F_gs_T = F_gs_T
        VFRayTracingRaw_T.F_gt_T = F_gt_T
        VFRayTracingRaw_T.F_gw_T = F_gw_T
        VFRayTracingRaw_T.F_ww_T = F_ww_T
        VFRayTracingRaw_T.F_wt_T = F_wt_T
        VFRayTracingRaw_T.F_wg_T = F_wg_T
        VFRayTracingRaw_T.F_ws_T = F_ws_T
        VFRayTracingRaw_T.F_sg_T = F_sg_T
        VFRayTracingRaw_T.F_sw_T = F_sw_T
        VFRayTracingRaw_T.F_st_T = F_st_T
        VFRayTracingRaw_T.F_tg_T = F_tg_T
        VFRayTracingRaw_T.F_tw_T = F_tw_T
        VFRayTracingRaw_T.F_ts_T = F_ts_T
        VFRayTracingRaw_T.F_tt_T = F_tt_T
        VFRayTracingRaw_T.F_pg = F_pg
        VFRayTracingRaw_T.F_ps = F_ps
        VFRayTracingRaw_T.F_pw = F_pw
        VFRayTracingRaw_T.F_pt = F_pt

        return F_gs_T,F_gt_T,F_gw_T,F_ww_T,F_wt_T,F_wg_T,F_ws_T,F_ts_T,F_tw_T,F_tt_T,F_tg_T,F_sg_T,F_sw_T,F_st_T,F_pg,\
               F_ps,F_pw,F_pt,VFRayTracingRaw_T

    def View_Factors_Geometry(self,H,W,a,ht,d,Person,OPTION_SURFACE,MCSampleSize,NRays):

        # Geometry specification
        h = H / W
        w = W / W

        pz = Person.PositionPz / W
        px = Person.PositionPx / W

        # Roof
        x1a = [0,1]
        z1a = [h,h]

        x1b = [1 + w,2 + w]
        z1b = [h,h]

        # Ground
        x2 = [1,1 + w]
        z2 = [0,0]

        # Wall 1
        x3 = [1,1]
        z3 = [h,0]

        # Wall 2
        x4 = [1 + w,1 + w]
        z4 = [0,h]

        # Sky
        x5 = [1,1 + w]
        z5 = [h,h]

        # Tree 1
        xc = 1 + d * w
        yc = ht * w
        r = a * w
        ang = numpy.arange(start=0,stop=2*numpy.pi,step=0.02)
        xt = [r * math.cos(ang[i]) for i in range(len(ang))]
        yt = [r * math.sin(ang[i]) for i in range(len(ang))]
        if r == 0:
            xc = 0
            yc = 0

        # Tree 2
        xc2 = 1 + w - d * w
        ang = numpy.arange(start=0,stop=2*numpy.pi,step=0.02)
        if r == 0:
            xc2 = 0
            yc = 0

        # Person
        xcp6 = 1 + px
        ycp6 = pz
        rp6 = 1 / 1000
        xp6 = [rp6 * math.cos(ang[i]) for i in range(len(ang))]
        yp6 = [rp6 * math.sin(ang[i]) for i in range(len(ang))]

        # Monte Carlo Parameters
        RandSZ = numpy.random.uniform(0,1,MCSampleSize)
        # Uniformly distributed "random" values in the interval [0,1]
        DeltaRays = numpy.arange(0,1+1/(NRays/2),1/(NRays/2))

        # polar angle (zenith)
        AnlgeDist = [math.asin(DeltaRays[i]) for i in range(len(DeltaRays))]
        # convert it to altitude/elevation angle in first quadrant
        RayAngleQ1 = [numpy.pi/2 - AnlgeDist[i] for i in range(len(AnlgeDist))][::-1]
        # Angle in second quadrant
        RayAngleQ2 = [numpy.pi / 2 + AnlgeDist[i] for i in range(len(AnlgeDist))]
        # for a horizontal planar surface
        RayAngle = RayAngleQ1[0:-1] + RayAngleQ2

        # Ray Angle is defined as the altitude angle on a horizontal surface
        # starting on the "right side" (first quadrat of coordinate system). It can
        # be used directly for the ground surface but needs to be shifted by +pi/2
        # and -pi/2 for the wall surfaces as the walls are vertical in our
        # coordinate system. It also needs to be shifted according to the
        # orientation of the tangent of the emitting point on the tree circle.

        # The emitting point needs to be slightly moved away from the surface.
        # Otherwise, it will be counted as crossing itself. stc defines how much a point is moved away from the surface
        # How far is the starting point away from the surface.
        stc = 10**(-10)

        # Vector definition
        if OPTION_SURFACE == 0:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View Factor from Wall-1
            # Randomly distributed emitting points
            YSv = [h * RandSZ[i] for i in range(len(RandSZ))]
            XSv = [(1+stc)*1 for i in range(len(YSv))]

            # Uniformly distributed emitting points
            RayAngle_array = numpy.array([RayAngle])
            dthe = numpy.ones((len(XSv), 1)) @ (RayAngle_array - numpy.pi / 2)

        elif OPTION_SURFACE == 1:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View Factor from Wall-2
            # Randomly distributed emitting points
            YSv = [h * RandSZ[i] for i in range(len(RandSZ))]
            XSv = [(1 + w - stc) * 1 for i in range(len(YSv))]

            # Uniformly distributed emitting points
            RayAngle_array = numpy.array([RayAngle])
            dthe = numpy.ones((len(XSv), 1)) @ (RayAngle_array + numpy.pi / 2)

        elif OPTION_SURFACE == 2:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View Factor from ground
            # Randomly distributed emitting points
            XSv = [1+w*RandSZ[i] for i in range(len(RandSZ))]
            YSv = [stc*1 for i in range(len(XSv))]

            # Uniformly distributed emitting points
            RayAngle_array = numpy.array([RayAngle])
            dthe = numpy.ones((len(XSv), 1)) @ (RayAngle_array)

        elif OPTION_SURFACE == 3:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View from Tree-1
            # Randomly distributed emitting points
            ang = [2*numpy.pi*RandSZ[i] for i in range(len(RandSZ))]

            # Uniformly distributed emitting points
            xt = [(r + stc) * math.cos(ang[i]) for i in range(len(ang))]
            yt = [(r + stc) * math.sin(ang[i]) for i in range(len(ang))]
            XSv = [xc + xt[i] for i in range(len(xt))]
            YSv = [yc + yt[i] for i in range(len(yt))]
            if r == 0:
                XSv[:] = [0 for k in range(len(XSv))]
                YSv[:] = [0 for k in range(len(YSv))]

            RayAngle_array = numpy.array([RayAngle])
            ang_array = numpy.array([ang])
            dthe = numpy.ones((len(XSv),1)) @ (RayAngle_array-numpy.pi/2) + ang_array.T

        elif OPTION_SURFACE == 4:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View from Tree-2
            # Randomly distributed emitting points
            ang = [2*numpy.pi*RandSZ[i] for i in range(len(RandSZ))]

            # Uniformly distributed emitting points
            xt = [(r + stc) * math.cos(ang[i]) for i in range(len(ang))]
            yt = [(r + stc) * math.sin(ang[i]) for i in range(len(ang))]
            XSv = [xc2 + xt[i] for i in range(len(xt))]
            YSv = [yc + yt[i] for i in range(len(yt))]
            if r == 0:
                XSv[:] = [0 for k in range(len(XSv))]
                YSv[:] = [0 for k in range(len(YSv))]

            RayAngle_array = numpy.array([RayAngle])
            ang_array = numpy.array([ang])
            dthe = numpy.ones((len(XSv), 1)) @ (RayAngle_array - numpy.pi / 2) + ang_array.T

        elif OPTION_SURFACE == 5:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View Factor from sky
            # Randomly distributed emitting points
            XSv = [(1+w*RandSZ[i]) for i in range(len(RandSZ))]
            YSv = [(h-stc)*1 for i in range(len(XSv))]

            # Uniformly distributed emitting points
            RayAngle_array = numpy.array([RayAngle])
            dthe = numpy.ones((len(XSv), 1)) @ (RayAngle_array + numpy.pi)

        elif OPTION_SURFACE == 6:
            print('OPTION_SURFACE: ', OPTION_SURFACE)
            # View from point for MRT
            ang = [2*numpy.pi*RandSZ[i] for i in range(len(RandSZ))]

            # Uniformly distributed emitting points
            xp6 = [(rp6+stc)*math.cos(ang[i]) for i in range(len(ang))]
            yp6 = [(rp6 + stc)*math.sin(ang[i]) for i in range(len(ang))]
            XSv = [xcp6 + xp6[i] for i in range(len(xp6))]
            YSv = [ycp6 + yp6[i] for i in range(len(yp6))]

            RayAngle_array = numpy.array([RayAngle])
            ang_array = numpy.array([ang])
            dthe = numpy.ones((len(XSv), 1)) @ (RayAngle_array - numpy.pi / 2) + ang_array.T

        # Parameters of the search
        # maximum ray length, maximum search distance
        dmax = numpy.sqrt(h ** 2 + w ** 2) + numpy.sqrt(h ** 2 + w ** 2) / 100
        # Search step size for tree detection
        sz = w / 1000
        # plots graph
        GRAPH = 0

        VG, VW1, VW2, VS, VT1, VT2  = self.ViewFactorsComputation(XSv, YSv, dmax, sz, dthe, GRAPH, x2, z2, x3,z3, x4, z4,
                                                                  xc, yc, r, xc2, x5, z5)

        return VG,VW1,VW2,VS,VT1,VT2

    def ViewFactorsComputation(self,XSv,YSv,dmax,sz,dthe,GRAPH,x2,z2,x3,z3,x4,z4,xc,yc,r,xc2,x5,z5):

        # pass of search
        spass = numpy.sqrt(2)*sz
        # search distance [m]
        SD = numpy.arange(start=spass,step=spass,stop=dmax)

        np = len(XSv)
        VGv = numpy.zeros(np)
        VW1v = numpy.zeros(np)
        VW2v = numpy.zeros(np)
        VSv = numpy.zeros(np)
        VT1v = numpy.zeros(np)
        VT2v = numpy.zeros(np)

        # For the number of emitting points
        for ii in range(np):
            # search angle [angular degree]
            Z = copy.copy(dthe[ii,:])

            XS = XSv[ii]
            YS = YSv[ii]
            VG = 0
            VW1 = 0
            VW2 = 0
            VS = 0
            VT1 = 0
            VT2 = 0

            # For the number of rays emitted from each emitting point
            for k in range(len(Z)):

                ## Check
                xp = [SD[ip]*numpy.cos(Z[k]) for ip in range(len(SD))]
                yp = [SD[ip]*numpy.sin(Z[k]) for ip in range(len(SD))]

                # Ground
                l1 = [x2[0],z2[0],x2[1],z2[1]]
                l2 = [XS,YS,XS+xp[-1],YS+yp[-1]]
                out = self.lineSegmentIntersect(l1, l2)
                Sfn = out.intAdjacencyMatrix
                xI = out.intMatrixX
                yI = out.intMatrixY
                if Sfn == 1:
                    # distance
                    D2 = numpy.sqrt(abs(xI-XS)**2 + abs(yI-YS)**2)
                else:
                    D2 = numpy.NaN

                # Wall-1
                l1 = [x3[0],z3[0],x3[1],z3[1]]
                l2 = [XS,YS,XS+xp[-1],YS+yp[-1]]
                out = self.lineSegmentIntersect(l1, l2)
                Sfn = out.intAdjacencyMatrix
                xI = out.intMatrixX
                yI = out.intMatrixY
                if Sfn == 1:
                    # distance
                    D3 = numpy.sqrt(abs(xI-XS)**2 + abs(yI-YS)**2)
                else:
                    D3 = numpy.NaN

                # Wall-2
                l1 = [x4[0], z4[0], x4[1], z4[1]]
                l2 = [XS, YS, XS + xp[-1], YS + yp[-1]]
                out = self.lineSegmentIntersect(l1, l2)
                Sfn = out.intAdjacencyMatrix
                xI = out.intMatrixX
                yI = out.intMatrixY
                if Sfn == 1:
                    # distance
                    D4 = numpy.sqrt(abs(xI - XS) ** 2 + abs(yI - YS) ** 2)
                else:
                    D4 = numpy.NaN

                # Sky
                l1 = [x5[0], z5[0], x5[1], z5[1]]
                l2 = [XS, YS, XS + xp[-1], YS + yp[-1]]
                out = self.lineSegmentIntersect(l1, l2)
                Sfn = out.intAdjacencyMatrix
                xI = out.intMatrixX
                yI = out.intMatrixY
                if Sfn == 1:
                    # distance
                    D5 = numpy.sqrt(abs(xI - XS) ** 2 + abs(yI - YS) ** 2)
                else:
                    D5 = numpy.NaN

                # Tree 1
                # Inside tree
                IC = numpy.zeros(len(xp))
                for i in range(len(xp)):
                    if (XS + xp[i] - xc)**2 + (YS + yp[i] - yc)**2 <= r**2:
                        IC[i] = 1
                    else:
                        IC[i] = 0
                if sum(IC) > 1:

                    sdi = min(numpy.where(IC == 1)[0])
                    DT1 = numpy.sqrt(abs(XS+xp[sdi] - XS) ** 2 + abs(YS+yp[sdi] - YS) ** 2)
                else:
                    DT1 = numpy.NaN

                # Tree 2
                # Inside tree
                IC = numpy.zeros(len(xp))
                for i in range(len(xp)):
                    if (XS + xp[i] - xc2) ** 2 + (YS + yp[i] - yc) ** 2 <= r ** 2:
                        IC[i] = 1
                    else:
                        IC[i] = 0
                if sum(IC) > 1:

                    sdi = min(numpy.where(IC == 1)[0])
                    DT2 = numpy.sqrt(abs(XS + xp[sdi] - XS) ** 2 + abs(YS + yp[sdi] - YS) ** 2)
                else:
                    DT2 = numpy.NaN

                # Assign a count for the surface that the ray is passing through
                # Ground  Wall 1 Wall 2  Tree 1  Tree 2 Sky
                md = min((imin for imin in [D2,D3,D4,DT1,DT2,D5] if not math.isnan(imin)))
                pmin = numpy.where([D2,D3,D4,DT1,DT2,D5] == md)[0]

                if numpy.isnan(md):
                    pass
                else:
                    if pmin == 0:
                        VG = VG + 1
                    elif pmin == 1:
                        VW1 = VW1 + 1
                    elif pmin == 2:
                        VW2 = VW2 + 1
                    elif pmin == 3:
                        VT1 = VT1 + 1
                    elif pmin == 4:
                        VT2 = VT2 + 1
                    else:
                        VS = VS + 1

            # Calculates the view factors for each emitting point
            VG = VG / len(Z)
            VW1 = VW1 / len(Z)
            VW2 = VW2 / len(Z)
            VS = VS / len(Z)
            VT1 = VT1 / len(Z)
            VT2 = VT2 / len(Z)

            # This should be 1
            Sum_view = sum([VG,VW1,VW2,VS,VT1,VT2])

            VGv[ii] = VG / Sum_view
            VW1v[ii] = VW1 / Sum_view
            VW2v[ii] = VW2 / Sum_view
            VSv[ii] = VS / Sum_view
            VT1v[ii] = VT1 / Sum_view
            VT2v[ii] = VT2 / Sum_view

        # Calcualtes the mean view factor of all the emitting points together
        VG = numpy.mean(VGv)
        VW1 = numpy.mean(VW1v)
        VW2 = numpy.mean(VW2v)
        VS = numpy.mean(VSv)
        VT1 = numpy.mean(VT1v)
        VT2 = numpy.mean(VT2v)

        return VG,VW1,VW2,VS,VT1,VT2

    def lineSegmentIntersect(self,XY1,XY2):

        n_rows_1 = 1 ## check
        n_cols_1 = len(XY1)
        n_rows_2 = 1 ## check
        n_cols_2 = len(XY2)

        if n_cols_1 != 4 or n_cols_2 != 4:
            print('Arguments must be a Nx4 matrices.')

        # Prepare matrices for vectorized computation of line intersection points.
        ## check
        X1 = XY1[0]
        X2 = XY1[2]
        Y1 = XY1[1]
        Y2 = XY1[3]
        ## check
        X3 = XY2[0]
        X4 = XY2[2]
        Y3 = XY2[1]
        Y4 = XY2[3]

        X4_X3 = (X4 - X3)
        Y1_Y3 = (Y1 - Y3)
        Y4_Y3 = (Y4 - Y3)
        X1_X3 = (X1 - X3)
        X2_X1 = (X2 - X1)
        Y2_Y1 = (Y2 - Y1)

        numerator_a = X4_X3 * Y1_Y3 - Y4_Y3 * X1_X3
        numerator_b = X2_X1 * Y1_Y3 - Y2_Y1 * X1_X3
        denominator = Y4_Y3 * X2_X1 - X4_X3 * Y2_Y1

        u_a = numerator_a / denominator
        u_b = numerator_b / denominator

        # Find the adjacency matrix A of intersecting lines.
        INT_X = X1 + X2_X1 * u_a
        INT_Y = Y1 + Y2_Y1 * u_a
        if (u_a >= 0) and (u_a <= 1) and (u_b >= 0) and (u_b <= 1):
            INT_B = 1
        else:
            INT_B = 0
        if denominator == 0:
            PAR_B = 1
        else:
            PAR_B = 0
        if (numerator_a == 0 and numerator_b == 0 and PAR_B == 1):
            COINC_B = 1
        else:
            COINC_B = 0

        # Arrange output.
        class out_Def():
            pass
        out = out_Def()
        out.intAdjacencyMatrix = INT_B
        out.intMatrixX = INT_X * INT_B
        out.intMatrixY = INT_Y * INT_B
        out.intNormalizedDistance1To2 = u_a
        out.intNormalizedDistance2To1 = u_b
        out.parAdjacencyMatrix = PAR_B
        out.coincAdjacencyMatrix = COINC_B

        return out